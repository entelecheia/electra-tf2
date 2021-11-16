#!/bin/bash

DATASET="ekon_pretok"
ROOT_DIR="/workspace/data/tbts/hydra/outputs/electra"

python3 create_tfrecords.py \
	rootdir=$ROOT_DIR \
	hydra.job.name=$DATASET \
	dataset=tfrecord/ekon_pretok
