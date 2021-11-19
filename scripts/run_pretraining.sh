#!/bin/bash

echo "Container nvidia build = " $NVIDIA_BUILD_ID

NUM_GPUS=${1:-4}
CONFIG_FILE_P1=${2:-"pretrain_ekon_pretok_base_p1"}
CONFIG_FILE_P2=${3:-"pretrain_ekon_pretok_base_p2"}
CODE_DIR=${4:-"/workspace/electra"}
LOG_DIR=${5:-"$CODE_DIR/logs"}
CONFIG_DIR=${5:-"$CODE_DIR/conf"}
RUN_P1=${6:-"false"}
RUN_P2=${7:-"true"}

mkdir -p $LOG_DIR

PREFIX=""
TEST_RESULT=$(awk 'BEGIN {print ('1' <= '${NUM_GPUS}')}')
if [ "$TEST_RESULT" == 1 ] ; then
    PREFIX="horovodrun -np $NUM_GPUS "
fi

if [ "$RUN_P1" == "true" ] ; then

   CMD=" electra.pretrain.run_pretraining"
   CMD+=" --config-dir $CONFIG_DIR"
   CMD+=" +run=${CONFIG_FILE_P1}"

   CMD="$PREFIX python3 -m $CMD"
   echo "Launch command: $CMD"

   DATESTAMP=`date +'%y%m%d%H%M%S'`
   LOGFILE=$LOG_DIR/$CONFIG_FILE_P1.$DATESTAMP.log
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

   echo "finished pretraining phase1"
fi

#Start Phase2
if [ "$RUN_P2" == "true" ] ; then

   CMD=" electra.pretrain.run_pretraining"
   CMD+=" --config-dir $CONFIG_DIR"
   CMD+=" +run=${CONFIG_FILE_P2}"

   CMD="$PREFIX python3 -m $CMD"
   echo "Launch command: $CMD"


   DATESTAMP=`date +'%y%m%d%H%M%S'`
   LOGFILE=$LOG_DIR/$CONFIG_FILE_P2.$DATESTAMP.log
   printf "Logs written to %s\n" "$LOGFILE"

   set -x
   if [ -z "$LOGFILE" ] ; then
      $CMD
   else
      (
      $CMD
      ) |& tee $LOGFILE
   fi

   echo "finished pretraining phase2"
fi
set +x
