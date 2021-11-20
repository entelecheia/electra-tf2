# coding=utf-8
# Copyright 2020 The Google Research Authors.
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for preparing pre-training data and supplying them to the model."""

import collections
import os

import numpy as np
import tensorflow as tf

from ..util import utils
from ..model import tokenization


def get_dataset(config, batch_size, num_cpu_threads=4, world_size=1, rank=0):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([config.max_seq_length], tf.int64),
    }

    input_files = []
    for input_pattern in config.pretrain_tfrecords.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))

    d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
    d = d.shard(num_shards=world_size, index=rank)
    d = d.repeat()
    d = d.shuffle(buffer_size=len(input_files), seed=config.seed, reshuffle_each_iteration=False)

    cycle_length = min(num_cpu_threads, len(input_files))
    d = d.interleave(
        tf.data.TFRecordDataset,
        cycle_length=cycle_length,
        deterministic=True)
    d = d.shuffle(buffer_size=100, seed=config.seed, reshuffle_each_iteration=False)

    d = d.map(lambda record: _decode_record(record, name_to_features))
    d = d.batch(batch_size)

    return d

def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t

    return example


# model inputs - it's a bit nicer to use a namedtuple rather than keep the
# features as a dict
Inputs = collections.namedtuple(
    "Inputs", ["input_ids", "input_mask", "segment_ids", "masked_lm_positions",
               "masked_lm_ids", "masked_lm_weights"])


def features_to_inputs(features):
    return Inputs(
        input_ids=features["input_ids"],
        input_mask=features["input_mask"],
        segment_ids=features["segment_ids"],
        masked_lm_positions=(features["masked_lm_positions"]
                             if "masked_lm_positions" in features else None),
        masked_lm_ids=(features["masked_lm_ids"]
                       if "masked_lm_ids" in features else None),
        masked_lm_weights=(features["masked_lm_weights"]
                           if "masked_lm_weights" in features else None),
    )


def get_updated_inputs(inputs, **kwargs):
    features = inputs._asdict()
    for k, v in kwargs.items():
        features[k] = v
    return features_to_inputs(features)


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
    if isinstance(tensor, np.ndarray) or isinstance(tensor, list):
        shape = np.array(tensor).shape
        if isinstance(expected_rank, six.integer_types):
            assert len(shape) == expected_rank
        elif expected_rank is not None:
            assert len(shape) in expected_rank
        return shape
    #
    # if name is None:
    #     name = tensor.name
    #
    # if expected_rank is not None:
    #     assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def gather_positions(sequence, positions):
    """Gathers the vectors at the specific positions over a minibatch.

  Args:
    sequence: A [batch_size, seq_length] or
        [batch_size, seq_length, depth] tensor of values
    positions: A [batch_size, n_positions] tensor of indices

  Returns: A [batch_size, n_positions] or
    [batch_size, n_positions, depth] tensor of the values at the indices
  """
    shape = get_shape_list(sequence, expected_rank=[2, 3])
    depth_dimension = (len(shape) == 3)
    if depth_dimension:
        B, L, D = shape
    else:
        B, L = shape
        D = 1
        sequence = tf.expand_dims(sequence, -1)
    position_shift = tf.expand_dims(L * tf.range(B), -1)
    flat_positions = tf.reshape(positions + position_shift, [-1])
    flat_sequence = tf.reshape(sequence, [B * L, D])
    gathered = tf.gather(flat_sequence, flat_positions)
    if depth_dimension:
        return tf.reshape(gathered, [B, -1, D])
    else:
        return tf.reshape(gathered, [B, -1])


def scatter_update(sequence, updates, positions):
    """Scatter-update a sequence.

  Args:
    sequence: A [batch_size, seq_len] or [batch_size, seq_len, depth] tensor
    updates: A tensor of size batch_size*seq_len(*depth)
    positions: A [batch_size, n_positions] tensor

  Returns: A tuple of two tensors. First is a [batch_size, seq_len] or
    [batch_size, seq_len, depth] tensor of "sequence" with elements at
    "positions" replaced by the values at "updates." Updates to index 0 are
    ignored. If there are duplicated positions the update is only applied once.
    Second is a [batch_size, seq_len] mask tensor of which inputs were updated.
  """
    shape = get_shape_list(sequence, expected_rank=[2, 3])
    depth_dimension = (len(shape) == 3)
    if depth_dimension:
        B, L, D = shape
    else:
        B, L = shape
        D = 1
        sequence = tf.expand_dims(sequence, -1)
    N = get_shape_list(positions)[1]

    shift = tf.expand_dims(L * tf.range(B), -1)
    flat_positions = tf.reshape(positions + shift, [-1, 1])
    flat_updates = tf.reshape(updates, [-1, D])
    updates = tf.scatter_nd(flat_positions, flat_updates, [B * L, D])
    updates = tf.reshape(updates, [B, L, D])

    flat_updates_mask = tf.ones([B * N], tf.int32)
    updates_mask = tf.scatter_nd(flat_positions, flat_updates_mask, [B * L])
    updates_mask = tf.reshape(updates_mask, [B, L])
    not_first_token = tf.concat([tf.zeros((B, 1), tf.int32),
                                 tf.ones((B, L - 1), tf.int32)], -1)
    updates_mask *= not_first_token
    updates_mask_3d = tf.expand_dims(updates_mask, -1)

    # account for duplicate positions
    if sequence.dtype == tf.float32:
        updates_mask_3d = tf.cast(updates_mask_3d, tf.float32)
        updates /= tf.maximum(1.0, updates_mask_3d)
    else:
        assert sequence.dtype == tf.int32
        updates = tf.math.floordiv(updates, tf.maximum(1, updates_mask_3d))
    updates_mask = tf.minimum(updates_mask, 1)
    updates_mask_3d = tf.minimum(updates_mask_3d, 1)

    updated_sequence = (((1 - updates_mask_3d) * sequence) +
                        (updates_mask_3d * updates))
    if not depth_dimension:
        updated_sequence = tf.squeeze(updated_sequence, -1)

    return updated_sequence, updates_mask


def _get_candidates_mask(inputs: Inputs, vocab,
                         disallow_from_mask=None):
    """Returns a mask tensor of positions in the input that can be masked out."""
    ignore_ids = [vocab["[SEP]"], vocab["[CLS]"], vocab["[MASK]"]]
    candidates_mask = tf.ones_like(inputs.input_ids, tf.bool)
    for ignore_id in ignore_ids:
        candidates_mask &= tf.not_equal(inputs.input_ids, ignore_id)
    candidates_mask &= tf.cast(inputs.input_mask, tf.bool)
    if disallow_from_mask is not None:
        candidates_mask &= ~disallow_from_mask
    return candidates_mask


def mask(config, inputs, mask_prob, proposal_distribution=1.0,
         disallow_from_mask=None, already_masked=None):
    """Implementation of dynamic masking. The optional arguments aren't needed for
    BERT/ELECTRA and are from early experiments in "strategically" masking out
    tokens instead of uniformly at random.

    Args:
      config: configure_pretraining.PretrainingConfig
      inputs: pretrain_data.Inputs containing input input_ids/input_mask
      mask_prob: percent of tokens to mask
      proposal_distribution: for non-uniform masking can be a [B, L] tensor
                             of scores for masking each position.
      disallow_from_mask: a boolean tensor of [B, L] of positions that should
                          not be masked out
      already_masked: a boolean tensor of [B, N] of already masked-out tokens
                      for multiple rounds of masking
    Returns: a pretrain_data.Inputs with masking added
    """
    # Get the batch size, sequence length, and max masked-out tokens
    N = config.max_predictions_per_seq
    B, L = get_shape_list(inputs.input_ids)

    # Find indices where masking out a token is allowed
    vocab = tokenization.ElectraTokenizer(
        config.vocab_file, do_lower_case=config.do_lower_case).get_vocab()
    candidates_mask = _get_candidates_mask(inputs, vocab, disallow_from_mask)

    # Set the number of tokens to mask out per example
    num_tokens = tf.cast(tf.reduce_sum(inputs.input_mask, -1), tf.float32)
    num_to_predict = tf.maximum(1, tf.minimum(
        N, tf.cast(tf.round(num_tokens * mask_prob), tf.int32)))
    masked_lm_weights = tf.cast(tf.sequence_mask(num_to_predict, N), tf.float32)
    if already_masked is not None:
        masked_lm_weights *= (1 - already_masked)

    # Get a probability of masking each position in the sequence
    candidate_mask_float = tf.cast(candidates_mask, tf.float32)
    sample_prob = (proposal_distribution * candidate_mask_float)
    sample_prob /= tf.reduce_sum(sample_prob, axis=-1, keepdims=True)

    # Sample the positions to mask out
    sample_prob = tf.stop_gradient(sample_prob)
    sample_logits = tf.math.log(sample_prob)
    masked_lm_positions = tf.random.categorical(
        sample_logits, N, dtype=tf.int32)
    masked_lm_positions *= tf.cast(masked_lm_weights, tf.int32)

    # Get the ids of the masked-out tokens
    shift = tf.expand_dims(L * tf.range(B), -1)
    flat_positions = tf.reshape(masked_lm_positions + shift, [-1, 1])
    masked_lm_ids = tf.gather_nd(tf.reshape(inputs.input_ids, [-1]),
                                 flat_positions)
    masked_lm_ids = tf.reshape(masked_lm_ids, [B, -1])
    masked_lm_ids *= tf.cast(masked_lm_weights, tf.int32)

    # Update the input ids
    replace_with_mask_positions = masked_lm_positions * tf.cast(
        tf.less(tf.random.uniform([B, N]), 0.85), tf.int32)
    inputs_ids, _ = scatter_update(
        inputs.input_ids, tf.fill([B, N], vocab["[MASK]"]),
        replace_with_mask_positions)

    return get_updated_inputs(
        inputs,
        input_ids=tf.stop_gradient(inputs_ids),
        masked_lm_positions=masked_lm_positions,
        masked_lm_ids=masked_lm_ids,
        masked_lm_weights=masked_lm_weights
    )


def unmask(inputs: Inputs):
    unmasked_input_ids, _ = scatter_update(
        inputs.input_ids, inputs.masked_lm_ids, inputs.masked_lm_positions)
    return get_updated_inputs(inputs, input_ids=unmasked_input_ids)


def sample_from_softmax(logits, disallow=None):
    if disallow is not None:
        logits -= 1000.0 * disallow
    uniform_noise = tf.random.uniform(
        get_shape_list(logits), minval=0, maxval=1)
    gumbel_noise = tf.cast(-tf.math.log(-tf.math.log(uniform_noise + 1e-9) + 1e-9), logits.dtype)
    return tf.one_hot(tf.argmax(tf.nn.softmax(logits + gumbel_noise), -1,
                                output_type=tf.int32), logits.shape[-1])


ENDC = "\033[0m"
COLORS = ["\033[" + str(n) + "m" for n in list(range(91, 97)) + [90]]
RED = COLORS[0]
BLUE = COLORS[3]
CYAN = COLORS[5]
GREEN = COLORS[1]


def print_tokens(inputs: Inputs, inv_vocab, updates_mask=None):
    """Pretty-print model inputs."""
    pos_to_tokid = {}
    for tokid, pos, weight in zip(
            inputs.masked_lm_ids[0], inputs.masked_lm_positions[0],
            inputs.masked_lm_weights[0]):
        if weight == 0:
            pass
        else:
            pos_to_tokid[pos] = tokid

    text = ""
    provided_update_mask = (updates_mask is not None)
    if not provided_update_mask:
        updates_mask = np.zeros_like(inputs.input_ids)
    for pos, (tokid, um) in enumerate(
            zip(inputs.input_ids[0], updates_mask[0])):
        token = inv_vocab[tokid]
        if token == "[PAD]":
            break
        if pos in pos_to_tokid:
            token = RED + token + " (" + inv_vocab[pos_to_tokid[pos]] + ")" + ENDC
            if provided_update_mask:
                assert um == 1
        else:
            if provided_update_mask:
                assert um == 0
        text += token + " "
    utils.log(utils.printable_text(text))


class PretrainingConfig(object):
    """Defines pre-training hyperparameters."""

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.seed = 42

        self.debug = False  # debug mode for quickly running things
        self.do_train = True  # pre-train ELECTRA
        self.do_eval = False  # evaluate generator/discriminator on unlabeled data
        # self.phase2 = False
        self.phase = 1

        # amp
        self.amp = True
        self.xla = True
        self.fp16_compression = False

        # optimizer type
        self.optimizer = 'adam'
        self.gradient_accumulation_steps = 1

        # lamb whitelisting for LN and biases
        self.skip_adaptive = False

        # loss functions
        self.electra_objective = True  # if False, use the BERT objective instead
        self.gen_weight = 1.0  # masked language modeling / generator loss
        self.disc_weight = 50.0  # discriminator loss
        self.mask_prob = 0.15  # percent of input tokens to mask out / replace

        # optimization
        self.learning_rate = 5e-4
        self.lr_decay_power = 0.5
        self.weight_decay_rate = 0.01
        self.num_warmup_steps = 10000
        self.opt_beta_1 = 0.878
        self.opt_beta_2 = 0.974
        self.end_lr = 0.0

        # training settings
        self.log_freq = 10
        self.skip_checkpoint = False
        self.save_checkpoints_steps = 1000
        self.num_train_steps = 1000000
        self.num_eval_steps = 100
        self.keep_checkpoint_max = 5  # maximum number of recent checkpoint files to keep;  change to 0 or None to keep all checkpoints
        self.restore_checkpoint = None
        self.load_weights = False

        # model settings
        self.model_size = "base"  # one of "small", "base", or "large"
        # override the default transformer hparams for the provided model size; see
        # modeling.BertConfig for the possible hparams and util.training_utils for
        # the defaults
        self.model_hparam_overrides = (
            kwargs["model_hparam_overrides"]
            if "model_hparam_overrides" in kwargs else {})
        self.embedding_size = None  # bert hidden size by default
        self.vocab_size = 30522  # number of tokens in the vocabulary
        self.do_lower_case = True  # lowercase the input?

        # generator settings
        self.uniform_generator = False  # generator is uniform at random
        self.shared_embeddings = True  # share generator/discriminator token embeddings?
        # self.untied_generator = True  # tie all generator/discriminator weights?
        self.generator_layers = 1.0  # frac of discriminator layers for generator
        self.generator_hidden_size = 0.25  # frac of discrim hidden size for gen
        self.disallow_correct = False  # force the generator to sample incorrect
        # tokens (so 15% of tokens are always
        # fake)
        self.temperature = 1.0  # temperature for sampling from generator

        # batch sizes
        self.max_seq_length = 128
        self.train_batch_size = 128
        self.eval_batch_size = 128

        self.results_dir = "results"
        self.json_summary = None

        self.wandb_group = 'electra'
        self.wandb_project = 'electra-pretraining'
        self.update(kwargs)
        # default locations of data files
        
        self.pretrain_tfrecords = os.path.join(
            "data", "pretrain_tfrecords/pretrain_data.tfrecord*")
        self.vocab_file = os.path.join("vocab", "vocab.txt")
        self.model_dir = os.path.join(self.results_dir, "models", model_name)
        self.checkpoints_dir = os.path.join(self.model_dir, "checkpoints")
        self.weights_dir = os.path.join(self.model_dir, "weights")
        self.results_txt = os.path.join(self.results_dir, "unsup_results.txt")
        self.results_pkl = os.path.join(self.results_dir, "unsup_results.pkl")
        self.log_dir = os.path.join(self.model_dir, "logs")

        self.max_predictions_per_seq = int((self.mask_prob + 0.005) *
                                           self.max_seq_length)

        # defaults for different-sized model
        if self.model_size == "base":
            self.embedding_size = 768
            self.hidden_size = 768
            self.num_hidden_layers = 12
            if self.hidden_size % 64 != 0:
                raise ValueError("Hidden size {} should be divisible by 64. Number of attention heads is hidden size {} / 64 ".format(self.hidden_size, self.hidden_size))	
            self.num_attention_heads = int(self.hidden_size / 64.)
        elif self.model_size == "large":
            self.embedding_size = 1024
            self.hidden_size = 1024
            self.num_hidden_layers = 24
            if self.hidden_size % 64 != 0:
                raise ValueError("Hidden size {} should be divisible by 64. Number of attention heads is hidden size {} / 64 ".format(self.hidden_size, self.hidden_size))
            self.num_attention_heads = int(self.hidden_size / 64.)
        else:
            raise ValueError("--model_size : 'base' and 'large supported only.")
        self.act_func = "gelu"
        self.hidden_dropout_prob = 0.1 
        self.attention_probs_dropout_prob = 0.1

        self.update(kwargs)

    def update(self, kwargs):
        for k, v in kwargs.items():
            if v is not None:
                self.__dict__[k] = v