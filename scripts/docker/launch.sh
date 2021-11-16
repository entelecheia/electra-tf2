#!/bin/bash
set -x
set -o allexport; source .env; set +o allexport

CMD=${1:-/bin/bash}

docker run -it --rm \
  --gpus '"device=0,1,2,3"' \
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
