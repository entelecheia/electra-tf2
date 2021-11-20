# coding=utf-8
# Copyright 2020 The Google Research Authors.
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
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

"""Pre-trains an ELECTRA model."""

import argparse
import collections
import json
import time
import datetime
import os
import wandb

import tensorflow as tf
import horovod.tensorflow as hvd
from horovod.tensorflow.compression import Compression
from ..util.gpu_affinity import set_affinity

from ..util import utils
import sys
from . import pretrain_utils
from .pretrain_utils import PretrainingConfig
from ..util.utils import get_rank, get_world_size, is_main_process, log, log_config, setup_logger, postprocess_dllog
from ..model.tokenization import ElectraTokenizer
from ..model.modeling import PretrainingModel
from ..model.optimization import create_optimizer, GradientAccumulator
from ..util.postprocess_pretrained_ckpt import extract_models_from_pretrained_ckpts
import dllogger

import hydra
from omegaconf import DictConfig, OmegaConf
from pprint import pprint


def metric_fn(config, metrics, eval_fn_inputs):
    """Computes the loss and accuracy of the model."""
    d = eval_fn_inputs
    metrics["masked_lm_accuracy"].update_state(
        y_true=tf.reshape(d["masked_lm_ids"], [-1]),
        y_pred=tf.reshape(d["masked_lm_preds"], [-1]),
        sample_weight=tf.reshape(d["masked_lm_weights"], [-1]))
    metrics["masked_lm_loss"].update_state(
        values=tf.reshape(d["mlm_loss"], [-1]),
        sample_weight=tf.reshape(d["masked_lm_weights"], [-1]))
    if config.electra_objective:
        metrics["sampled_masked_lm_accuracy"].update_state(
            y_true=tf.reshape(d["masked_lm_ids"], [-1]),
            y_pred=tf.reshape(d["sampled_tokids"], [-1]),
            sample_weight=tf.reshape(d["masked_lm_weights"], [-1]))
        if config.disc_weight > 0:
            metrics["disc_loss"].update_state(d["disc_loss"])
            try:
                metrics["disc_auc"].update_state(
                d["disc_labels"] * d["input_mask"],
                d["disc_probs"] * tf.cast(d["input_mask"], tf.float32))
            except:
                pass
            metrics["disc_accuracy"].update_state(
                y_true=d["disc_labels"], y_pred=d["disc_preds"],
                sample_weight=d["input_mask"])
            metrics["disc_precision"].update_state(
                y_true=d["disc_labels"], y_pred=d["disc_preds"],
                sample_weight=d["disc_preds"] * d["input_mask"])
            metrics["disc_recall"].update_state(
                y_true=d["disc_labels"], y_pred=d["disc_preds"],
                sample_weight=d["disc_labels"] * d["input_mask"])
    return metrics

@tf.function
def train_one_step(config, model, optimizer, features, accumulator, first_step, take_step, clip_norm=1.0):

    #Forward and Backward pass
    with tf.GradientTape() as tape:
        total_loss, eval_fn_inputs = model(features, is_training=True)
        unscaled_loss = tf.stop_gradient(total_loss)
        if config.amp:
            total_loss = optimizer.get_scaled_loss(total_loss)
   
    #Backpropogate gradients
    #tape = hvd.DistributedGradientTape(
    #    tape, sparse_as_dense=True,
    #    compression=Compression.fp16 if config.amp and config.fp16_compression else Compression.none)
    gradients = tape.gradient(total_loss, model.trainable_variables)

    #Get unscaled gradients if AMP
    if config.amp:
        gradients = optimizer.get_unscaled_gradients(gradients)

    #Accumulate gradients
    accumulator(gradients)
    #Need to call apply_gradients on very first step irrespective of gradient accumulation
    #This is required for the optimizer to build it's states
    if first_step or take_step:
        #All reduce and Clip the accumulated gradients
        allreduced_accumulated_gradients = [None if g is None else hvd.allreduce(g / tf.cast(config.gradient_accumulation_steps, g.dtype),
                                compression=Compression.fp16 if config.amp and config.fp16_compression else Compression.none)
                                for g in accumulator.gradients]
        (clipped_accumulated_gradients, _) = tf.clip_by_global_norm(allreduced_accumulated_gradients, clip_norm=clip_norm)
        #Weight update
        optimizer.apply_gradients(zip(clipped_accumulated_gradients, model.trainable_variables))
        accumulator.reset()

    #brodcast model weights after first train step
    if first_step:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(optimizer.variables(), root_rank=0)

    return unscaled_loss, eval_fn_inputs

def main(args, config, e2e_start_time): 
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    # Set up tensorflow
    hvd.init()

    # DLLogger
    setup_logger(args)
    # wandb init
    wandb.init(config=config, group=config.wandb_group, project=config.wandb_project, dir=config.wandb_dir)

    set_affinity(hvd.local_rank())
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    tf.config.optimizer.set_jit(config.xla)
    #tf.config.optimizer.set_experimental_options({"auto_mixed_precision": config.amp})

    if config.amp:
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16", loss_scale="dynamic")
        tf.keras.mixed_precision.experimental.set_policy(policy)
        print('Compute dtype: %s' % policy.compute_dtype)  # Compute dtype: float16
        print('Variable dtype: %s' % policy.variable_dtype)  # Variable dtype: float32

    #tf.random.set_seed(config.seed)

    # Set up config cont'
    if config.load_weights and config.restore_checkpoint:
        raise ValueError("`load_weights` and `restore_checkpoint` should not be on at the same time.")
    if config.phase > 1 and not config.restore_checkpoint:
    # if config.phase2 and not config.restore_checkpoint:
        raise ValueError("`phase > 1` cannot be used without `restore_checkpoint`.")
    utils.heading("Config:")
    log_config(config)

    # Save pretrain configs
    pretrain_config_json = os.path.join(config.checkpoints_dir, 'pretrain_config.json')
    if is_main_process():
        utils.write_json(config.__dict__, pretrain_config_json)
        log("Configuration saved in {}".format(pretrain_config_json))

    # Set up model
    model = PretrainingModel(config)

    # Set up metrics
    metrics = dict()
    metrics["train_perf"] = tf.keras.metrics.Mean(name="train_perf")
    metrics["total_loss"] = tf.keras.metrics.Mean(name="total_loss")
    metrics["masked_lm_accuracy"] = tf.keras.metrics.Accuracy(name="masked_lm_accuracy")
    metrics["masked_lm_loss"] = tf.keras.metrics.Mean(name="masked_lm_loss")
    if config.electra_objective:
        metrics["sampled_masked_lm_accuracy"] = tf.keras.metrics.Accuracy(name="sampled_masked_lm_accuracy")
        if config.disc_weight > 0:
            metrics["disc_loss"] = tf.keras.metrics.Mean(name="disc_loss")
            metrics["disc_auc"] = tf.keras.metrics.AUC(name="disc_auc")
            metrics["disc_accuracy"] = tf.keras.metrics.Accuracy(name="disc_accuracy")
            metrics["disc_precision"] = tf.keras.metrics.Accuracy(name="disc_precision")
            metrics["disc_recall"] = tf.keras.metrics.Accuracy(name="disc_recall")

    # Set up tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(config.log_dir, current_time,
                                 'train_' + str(get_rank()) + '_of_' + str(get_world_size()))
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Set up dataset
    dataset = pretrain_utils.get_dataset(
        config, config.train_batch_size, world_size=get_world_size(), rank=get_rank())
    train_iterator = iter(dataset)

    # Set up optimizer
    optimizer = create_optimizer(
        init_lr=config.learning_rate,
        num_train_steps=config.num_train_steps,
        num_warmup_steps=config.num_warmup_steps,
        weight_decay_rate=config.weight_decay_rate,
        optimizer=config.optimizer,
        skip_adaptive=config.skip_adaptive,
        power=config.lr_decay_power,
        beta_1=config.opt_beta_1,
        beta_2=config.opt_beta_2,
        end_lr=config.end_lr)
        
    accumulator = GradientAccumulator()
    if config.amp:
        optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")

    # Set up model checkpoint
    checkpoint = tf.train.Checkpoint(
        step=tf.Variable(0), phase=tf.Variable(1), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, config.checkpoints_dir, max_to_keep=config.keep_checkpoint_max)
    if config.restore_checkpoint and config.restore_checkpoint != "latest":
        checkpoint.restore(config.restore_checkpoint)
        log(" ** Restored model checkpoint from {}".format(config.restore_checkpoint))
    elif config.restore_checkpoint and config.restore_checkpoint == "latest" and manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        log(" ** Restored model checkpoint from {}".format(manager.latest_checkpoint))
    elif config.load_weights:
        model.generator(model.generator.dummy_inputs)
        model.discriminator(model.discriminator.dummy_inputs)
        model.generator.load_weights(os.path.join(config.weights_dir, 'generator', 'tf_model.h5'))
        model.discriminator.load_weights(os.path.join(config.weights_dir, 'discriminator', 'tf_model.h5'))
    else:
        log(" ** Initializing from scratch.")

    restore_iterator = bool(config.restore_checkpoint) and config.restore_checkpoint == "latest"
    # Initialize global step for phase2
    if config.phase > 1 and int(checkpoint.phase) == 1:
        optimizer.iterations.assign(0)
        checkpoint.step.assign(0)
        checkpoint.phase.assign(config.phase)
        # checkpoint.phase2.assign(True)
        restore_iterator = False
    if int(checkpoint.phase) > 1:
        manager = tf.train.CheckpointManager(
            checkpoint, config.checkpoints_dir,
            checkpoint_name=f'ckpt-p{config.phase}',
            max_to_keep=config.keep_checkpoint_max)

    # Set up iterator checkpoint
    iter_checkpoint = tf.train.Checkpoint(
        train_iterator=train_iterator, world_size=tf.Variable(get_world_size()), rank=tf.Variable(get_rank()))
    iter_manager = tf.train.CheckpointManager(
        iter_checkpoint,
        os.path.join(config.checkpoints_dir, 'iter_ckpt_rank_' + '{:02}'.format(get_rank())),
        checkpoint_name='iter_ckpt_rank_' + '{:02}'.format(get_rank()),
        max_to_keep=config.keep_checkpoint_max)
    if restore_iterator and iter_manager.latest_checkpoint:
        ckpt_world_size = tf.train.load_variable(
            iter_manager.latest_checkpoint, 'world_size/.ATTRIBUTES/VARIABLE_VALUE')
        if ckpt_world_size == get_world_size():
            iter_checkpoint.restore(iter_manager.latest_checkpoint)
            log(" ** Restored iterator checkpoint from {}".format(iter_manager.latest_checkpoint), all_rank=True)

    utils.heading("Running training")
    accumulator.reset()
    train_start, start_step = time.time(), int(checkpoint.step) - 1
    local_step = 0
    saved_ckpt = False
    while int(checkpoint.step) <= config.num_train_steps:
        saved_ckpt = False
        step = int(checkpoint.step)
        features = next(train_iterator)
        iter_start = time.time()

        # if step == 200: tf.profiler.experimental.start(logdir=train_log_dir)
        total_loss, eval_fn_inputs = train_one_step(config, model, optimizer, features, accumulator,
                                                    local_step==1, take_step=local_step % config.gradient_accumulation_steps == 0)
        # if step == 300: tf.profiler.experimental.stop()

        metrics["train_perf"].update_state(
            config.train_batch_size * get_world_size() / (time.time() - iter_start))
        metrics["total_loss"].update_state(values=total_loss)
        metric_fn(config, metrics, eval_fn_inputs)

        if (step % config.log_freq == 0) and (local_step % config.gradient_accumulation_steps == 0):
            log_info_dict = {k:float(v.result().numpy() * 100) if "accuracy" in k else float(v.result().numpy()) for k, v in metrics.items()}
            log_info_dict["lr"] = float(optimizer._optimizer._decayed_lr('float32').numpy())
            dllogger.log(step=(step,), data=log_info_dict, verbosity=0)
            wandb.log(log_info_dict, step=step)
            log('Step:{step:6d}, Loss:{total_loss:10.6f}, Gen_loss:{masked_lm_loss:10.6f}, Disc_loss:{disc_loss:10.6f}, Gen_acc:{masked_lm_accuracy:6.2f}, '
                'Disc_acc:{disc_accuracy:6.2f}, Perf:{train_perf:4.0f}, Loss Scaler: {loss_scale}, Elapsed: {elapsed}, ETA: {eta}, '.format(
                step=step, **log_info_dict,
                loss_scale=optimizer.loss_scale if config.amp else 1,
                elapsed=utils.get_readable_time(time.time() - train_start),
                eta=utils.get_readable_time(
                    (time.time() - train_start) / (step - start_step) * (config.num_train_steps - step))),
                all_rank=True)

            with train_summary_writer.as_default():
                for key, m in metrics.items():
                    tf.summary.scalar(key, m.result(), step=step)

            if int(checkpoint.step) < config.num_train_steps:
                for m in metrics.values():
                    m.reset_states()

        #Print allreduced metrics on the last step
        if int(checkpoint.step) == config.num_train_steps and (local_step % config.gradient_accumulation_steps == 0):
            log_info_dict = {k:float(hvd.allreduce(v.result()).numpy() * 100) if "accuracy" in k else float(hvd.allreduce(v.result()).numpy()) for k, v in metrics.items()}
            wandb.log(log_info_dict)
            log_info_dict["training_sequences_per_second"] = log_info_dict["train_perf"]
            log_info_dict["final_loss"] = log_info_dict["total_loss"]
            log_info_dict["e2e_train_time"] = time.time() - e2e_start_time
            dllogger.log(step=(), data=log_info_dict, verbosity=0)
            log('<FINAL STEP METRICS> Step:{step:6d}, Loss:{total_loss:10.6f}, Gen_loss:{masked_lm_loss:10.6f}, Disc_loss:{disc_loss:10.6f}, Gen_acc:{masked_lm_accuracy:6.2f}, '
                'Disc_acc:{disc_accuracy:6.2f}, Perf:{train_perf:4.0f},'.format(
                step=step, **log_info_dict),
                all_rank=False)

        if local_step % config.gradient_accumulation_steps == 0:
            checkpoint.step.assign(int(optimizer.iterations))
        
        local_step += 1
        if not config.skip_checkpoint and (local_step % (config.save_checkpoints_steps * config.gradient_accumulation_steps) == 0):
            saved_ckpt = True
            if is_main_process():
                save_path = manager.save(checkpoint_number=step)
                log(" ** Saved model checkpoint for step {}: {}".format(step, save_path))
            iter_save_path = iter_manager.save(checkpoint_number=step)
            log(" ** Saved iterator checkpoint for step {}: {}".format(step, iter_save_path), all_rank=True)

    step = (int(checkpoint.step) - 1)
    dllogger.flush()
    if not config.skip_checkpoint and not saved_ckpt:
        if is_main_process():
            save_path = manager.save(checkpoint_number=step)
            log(" ** Saved model checkpoint for step {}: {}".format(step, save_path))
        iter_save_path = iter_manager.save(checkpoint_number=step)
        log(" ** Saved iterator checkpoint for step {}: {}".format(step, iter_save_path), all_rank=True)

    return args


@hydra.main(config_path="conf", config_name="config")
def hydra_main(cfg: DictConfig):

    # Pretty print config using Rich library
    if cfg.get("print_config"):
        print('## hydra configuration ##')
        print(OmegaConf.to_yaml(cfg))

    if cfg.get("print_resolved_config"):
        args = OmegaConf.to_container(cfg, resolve=True)
        print('## hydra configuration resolved ##')
        pprint(args)
        print()

    log(f"Current working directory : {os.getcwd()}")
    start_time = time.time()
    config = PretrainingConfig(**cfg.training)
    print(config)
    os.makedirs(config.results_dir, exist_ok=True)
    args = main(cfg.training, config, start_time)
    log("Total Time:{:.4f}".format(time.time() - start_time))
    if is_main_process():
        postprocess_dllog(args)
        if args.archive_after_training:
            log(f"Archiviing checkpoints from {config.checkpoints_dir} to {config.archive_dir}")
            os.makedirs(config.archive_dir, exist_ok=True)
            os.system(f"cp -rf {config.checkpoints_dir} {config.archive_dir}")
            extract_models_from_pretrained_ckpts(config)


if __name__ == "__main__":
    hydra_main()