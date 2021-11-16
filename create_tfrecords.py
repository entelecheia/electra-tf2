# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

import os
import subprocess
from timer import elapsed_timer
from omegaconf import OmegaConf


class TFRecords:
	def __init__(self, **kwargs):
		self.args = OmegaConf.create(kwargs)

	def run(self):
		with elapsed_timer(format_time=True) as elapsed:
			create_tfrecords(self.args)
			print(f"\n >>> Elapsed time: {elapsed()} <<< ")

def create_tfrecords(args):
    output_dir = args.output_dir

    print('Working Directory:', output_dir)
    print('Dataset Name:', args.dataset)

    sharded_dir = args.sharded_dir
    tfrecord_dir = args.tfrecord_dir
    print('tfrecord Directory:', tfrecord_dir)

    if not os.path.exists(tfrecord_dir + "/" + args.dataset):
        os.makedirs(tfrecord_dir + "/" + args.dataset)
    if args.vocab_file is None:
        args.vocab_file = os.path.join(output_dir, "vocab.txt")

    _dirs = []
    if args.n_training_shards > 0:
        _dirs.append('train')
    if args.n_test_shards > 0:
        _dirs.append('test')

    for _dir in _dirs:
        electra_preprocessing_command = 'python ' + args.runtime_dir + '/build_pretraining_dataset.py'
        electra_preprocessing_command += ' --corpus-dir=' + sharded_dir + '/' + args.dataset + '/' + _dir
        electra_preprocessing_command += ' --output-dir=' + tfrecord_dir + '/' + args.dataset + '/' + _dir
        electra_preprocessing_command += ' --vocab-file=' + args.vocab_file
        electra_preprocessing_command += ' --do-lower-case' if args.do_lower_case else ' --no-lower-case'
        electra_preprocessing_command += ' --max-seq-length=' + str(args.max_seq_length)
        electra_preprocessing_command += ' --num-processes=' + str(args.n_processes)
        electra_preprocessing_command += ' --num-out-files=' + str(args.n_training_shards) if _dir == 'train' \
            else ' --num-out-files=' + str(args.n_test_shards)
        electra_preprocessing_process = subprocess.Popen(electra_preprocessing_command, shell=True)

        electra_preprocessing_process.wait()

