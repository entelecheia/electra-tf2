# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
"""Convert ELECTRA checkpoint."""


import os
import argparse
import logging

import torch

from transformers import ElectraConfig, ElectraForMaskedLM, ElectraForPreTraining, load_tf_weights_in_electra


logging.basicConfig(level=logging.INFO)


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path, discriminator_or_generator):
    # Initialise PyTorch model
    config = ElectraConfig.from_json_file(config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))

    if discriminator_or_generator == "discriminator":
        model = ElectraForPreTraining(config)
    elif discriminator_or_generator == "generator":
        model = ElectraForMaskedLM(config)
    else:
        raise ValueError("The discriminator_or_generator argument should be either 'discriminator' or 'generator'")

    # Load weights from tf checkpoint
    load_tf_weights_in_electra(
        model, config, tf_checkpoint_path, discriminator_or_generator=discriminator_or_generator
    )

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained model. \n"
        "This specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--discriminator_or_generator",
        default=None,
        type=str,
        required=True,
        help="Whether to export the generator or the discriminator. Should be a string, either 'discriminator' or "
        "'generator'.",
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(
        args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path, args.discriminator_or_generator
    )


# parser = argparse.ArgumentParser()

# parser.add_argument("--tf_ckpt_path", type=str, default="koelectra-small-tf")
# parser.add_argument("--pt_discriminator_path", type=str, default="koelectra-small-discriminator")
# parser.add_argument("--pt_generator_path", type=str, default="koelectra-small-generator")

# args = parser.parse_args()

# convert_tf_checkpoint_to_pytorch(tf_checkpoint_path=args.tf_ckpt_path,
#                                  config_file=os.path.join(args.pt_discriminator_path, "config.json"),
#                                  pytorch_dump_path=os.path.join(args.pt_discriminator_path, "pytorch_model.bin"),
#                                  discriminator_or_generator="discriminator")

# convert_tf_checkpoint_to_pytorch(tf_checkpoint_path=args.tf_ckpt_path,
#                                  config_file=os.path.join(args.pt_generator_path, "config.json"),
#                                  pytorch_dump_path=os.path.join(args.pt_generator_path, "pytorch_model.bin"),
#                                  discriminator_or_generator="generator")