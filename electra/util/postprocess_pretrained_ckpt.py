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

import argparse
import collections
import json
import os
import glob
from pathlib import Path

import tensorflow as tf

from .utils import log, heading
from ..pretrain.pretrain_utils import PretrainingConfig
from ..model.modeling import PretrainingModel

import hydra
from omegaconf import DictConfig, OmegaConf
from pprint import pprint


def from_pretrained_ckpt(pretrained_checkpoint, output_dir, amp=False):
    config = PretrainingConfig(
        model_name='postprocessing',
        data_dir='postprocessing',
        generator_hidden_size=0.3333333,
    )

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    if amp:
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16", loss_scale="dynamic")
        tf.keras.mixed_precision.experimental.set_policy(policy)
        print('Compute dtype: %s' % policy.compute_dtype)  # Compute dtype: float16
        print('Variable dtype: %s' % policy.variable_dtype)  # Variable dtype: float32

    # Set up model
    model = PretrainingModel(config)

    # Load checkpoint
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), model=model)
    checkpoint.restore(pretrained_checkpoint).expect_partial()
    log(" ** Restored from {} at step {}".format(pretrained_checkpoint, int(checkpoint.step) - 1))

    disc_dir = os.path.join(output_dir, 'discriminator')
    gen_dir = os.path.join(output_dir, 'generator')

    heading(" ** Saving discriminator")
    model.discriminator(model.discriminator.dummy_inputs)
    model.discriminator.save_pretrained(disc_dir)

    heading(" ** Saving generator")
    model.generator(model.generator.dummy_inputs)
    model.generator.save_pretrained(gen_dir)


def extract_models_from_pretrained_ckpts(args):
    print(f"Extracting discriminators and generators from {args.checkpoints_dir} to {args.output_dir}")
    for f in glob.glob(f'{args.checkpoints_dir}/*.index'):
        ckpt = Path(f).stem
        output_dir = f'{args.output_dir}/{ckpt}'
        from_pretrained_ckpt(f, output_dir, amp=args.amp)


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

    config = PretrainingConfig(**cfg.training)
    print(config)
    extract_models_from_pretrained_ckpts(config)


if __name__ == "__main__":
    hydra_main()