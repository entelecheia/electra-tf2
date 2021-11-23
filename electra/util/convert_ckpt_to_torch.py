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
import glob
import logging
import json
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pprint import pprint

from transformers import ElectraConfig, ElectraForMaskedLM, ElectraForPreTraining, load_tf_weights_in_electra

logging.basicConfig(level=logging.INFO)


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path, model_type):
    # Initialise PyTorch model
    config = ElectraConfig.from_json_file(config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))

    if model_type == "discriminator":
        model = ElectraForPreTraining(config)
    elif model_type == "generator":
        model = ElectraForMaskedLM(config)
    else:
        raise ValueError("The discriminator_or_generator argument should be either 'discriminator' or 'generator'")

    # Load weights from tf checkpoint
    load_tf_weights_in_electra(
        model, config, tf_checkpoint_path, discriminator_or_generator=model_type
    )

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)

def convert_ckpt_to_pytorch(cfg):
    args=cfg.training
    if args.archive_after_training:
        print(f"Archiviing checkpoints from {args.checkpoints_dir} to {args.archive_dir}")
        os.makedirs(args.archive_dir, exist_ok=True)
        os.system(f"cp -rf {args.checkpoints_dir}/* {args.archive_dir}")
    print(f'Converting tensorflow checkpoints to pytoch')
    os.makedirs(args.torch_output_dir, exist_ok=True)
    ckpt_file = f'{args.archive_dir}/checkpoint'
    org_contents = open(ckpt_file, 'r').read()
    for f in glob.glob(f'{args.archive_dir}/*.index'):
        ckpt = Path(f).stem
        print('processing checkpoint-{}'.format(ckpt))
        ckpt_contents = f'model_checkpoint_path: "{ckpt}"\n'
        open(ckpt_file, 'w').write(ckpt_contents)
        model_config = OmegaConf.to_container(cfg.model, resolve=True)
        tokenizer_config = OmegaConf.to_container(cfg.tokenizer, resolve=True)
        for model_type in ['discriminator', 'generator']:
            torch_config = model_config[model_type]
            output_dir = f'{args.torch_output_dir}/{ckpt}/{model_type}'
            os.makedirs(output_dir, exist_ok=True)
            # create config
            config_file = f'{output_dir}/config.json'
            with open(config_file, 'w') as f:
                json.dump(torch_config, f, ensure_ascii=False, indent=4)
            with open(f'{output_dir}/tokenizer_config.json', 'w') as f:
                json.dump(tokenizer_config, f, ensure_ascii=False, indent=4) 
            # copy vocab
            os.system(f"cp -f {args.vocab_file} {output_dir}/vocab.txt")
            # model path
            pytorch_dump_path = f'{output_dir}/pytorch_model.bin'
            print(f'Saving {model_type} to {pytorch_dump_path}')
            convert_tf_checkpoint_to_pytorch(
                args.archive_dir, config_file, pytorch_dump_path, model_type
            )
    # restore original contents
    open(ckpt_file, 'w').write(org_contents)


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

    convert_ckpt_to_pytorch(cfg) 
 

if __name__ == "__main__":
    hydra_main()


