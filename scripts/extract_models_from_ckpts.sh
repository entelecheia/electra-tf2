#!/usr/bin/env bash
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

model_name=${1:-"pathmed"}
model_size=${2:-"base"}
models_dir=${3:-"/workspace/data/tbts/models/electra"}
ckpts_dir=$models_dir/$model_name/$model_size/checkpoints
output_dir=$models_dir/$model_name/$model_size/models

echo "Saving discriminators from $ckpts_dir to $output_dir"

for f in $ckpts_dir/*.index; do
    ckpt=${f%.*}
    ckpt_output_dir=$output_dir/$(basename $ckpt)
    mkdir -p $ckpt_output_dir
    echo "==================================== START $ckpt ===================================="
    python -m electra.util.postprocess_pretrained_ckpt --pretrained_checkpoint=$ckpt --output_dir=$ckpt_output_dir --amp
    echo "====================================  END $ckpt_output_dir  ====================================";
done
