#!/bin/bash
set -x
set -o allexport; source .env; set +o allexport

NV_VISIBLE_DEVICES=${1:-"2,3,4,5,6,7"}
CMD=${2:-/bin/bash}

docker run -it --rm \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$NV_VISIBLE_DEVICES \
  --net=host \
  --ipc=host \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e LD_LIBRARY_PATH='/workspace/install/lib/' \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -v $PWD:/workspace/electra \
  -v /home/yjlee/workspace/projects:/workspace/electra/projects \
  -v /home/yjlee/workspace/data:/workspace/electra/data \
  -v /home/yjlee/workspace/projects:/workspace/projects \
  -v /home/yjlee/workspace/data:/workspace/data \
  electra:20.07-tf2-py3  $CMD
