#!/bin/bash

echo "Container nvidia build = " $NVIDIA_BUILD_ID

NUM_GPUS=${1:-5}
PHASE=${2:-1}
CONFIG_FILE=${3:-"pretrain_ekon_pretok2_base"}
CODE_DIR=${4:-"/workspace/electra"}
LOG_DIR=${5:-"$CODE_DIR/logs"}
CONFIG_DIR=${5:-"$CODE_DIR/conf"}
PHASE_CONFIG_FILE=${CONFIG_FILE}_p${PHASE}

mkdir -p $LOG_DIR

PREFIX=""
TEST_RESULT=$(awk 'BEGIN {print ('1' <= '${NUM_GPUS}')}')
if [ "$TEST_RESULT" == 1 ] ; then
    PREFIX="horovodrun -np $NUM_GPUS "
fi

CMD=" electra.pretrain.run_pretraining"
CMD+=" --config-dir $CONFIG_DIR"
CMD+=" +run=${PHASE_CONFIG_FILE}"

CMD="$PREFIX python3 -m $CMD"
echo "Launch command: $CMD"

DATESTAMP=`date +'%y%m%d%H%M%S'`
LOGFILE=$LOG_DIR/$PHASE_CONFIG_FILE.$DATESTAMP.log
printf "Logs written to %s\n" "$LOGFILE"

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
   $CMD
   ) |& tee $LOGFILE
fi

set +x

echo "finished pretraining phase${PHASE}"

set +x
