#!/usr/bin/env bash

set -x
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

GPUS=$1
GPUS_PER_NODE=${GPUS}
name=$2
CPUS_PER_TASK=8

PY_ARGS=${@:3}

srun -p shlab-perceptionx \
    --quotatype=auto \
    --mpi=pmi2 \
    --job-name=${name} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${PY_ARGS} 2>&1|tee log/train-$now.log &
    # --quotatype=spot \
    # -w SH-IDC1-10-140-0-227 \