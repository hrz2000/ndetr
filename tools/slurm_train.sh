#!/usr/bin/env bash

set -x
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

CONFIG=$1
GPUS=$2
GPUS_PER_NODE=$2
CPUS_PER_TASK=24 # 24
SRUN_ARGS=${SRUN_ARGS:-""}
job_name=$3
PY_ARGS=${@:4}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p shlab-perceptionx \
    --quotatype=reserved \
    --mpi=pmi2 \
    --job-name=${job_name} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS} 2>&1|tee log/train-$now.log &
    # -w SH-IDC1-10-140-0-227 \
    # --quotatype=spot \