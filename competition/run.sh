#!/usr/bin/env bash

# args: host roundnum gamenum
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR
conda deactivate &> /dev/null
source $DIR/.bashrc
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH
cd $DIR/rv09ivin-bughouse/code/
LOG_DIR=$DIR/logs
mkdir -p $LOG_DIR
DATE=`date +%Y-%m-%d_%H-%M-%S`
TOURNAMENT="$(printf "%03d" ${2})_${DATE}"
mkdir -p "${LOG_DIR}/${TOURNAMENT}"
OUT_FILE="${LOG_DIR}/${TOURNAMENT}/${TOURNAMENT}_$(printf "%03d" ${3})_${4}.log"
export CUDA_VISIBLE_DEVICES="-1"
python3 -u run_cecp_websocket_competition.py $1 > $OUT_FILE 2>&1
