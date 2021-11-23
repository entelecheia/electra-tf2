#!/usr/bin/env bash

PHASE=${1:-1}
CONFIG_FILE=${2:-"ekon_pretok2_pretrain_base"}
CONFIG_DIR=${3:-"/workspace/electra/conf"}
PHASE_CONFIG_FILE=${CONFIG_FILE}_p${PHASE}


CMD=" electra.util.convert_ckpt_to_torch"
CMD+=" --config-dir $CONFIG_DIR"
CMD+=" +run=${PHASE_CONFIG_FILE}"

CMD="python3 -m $CMD"
echo "Launch command: $CMD"
$CMD
