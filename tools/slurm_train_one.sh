#!/usr/bin/env bash

set -x
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

CONFIG=$1
GPUS=$2
GPUS_PER_NODE=$2
CPUS_PER_TASK=24 # 24
job_name=$3
PY_ARGS=${@:4}

# export OMP_NUM_THREADS=4
# export MKL_NUM_THREADS=4
# /mnt/cache/herunze/miniconda3/envs/plant/lib/python3.8/site-packages/mmdet/utils/setup_env.py:38: UserWarning: Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being 
# overloaded, please further tune the variable for optimal performance in your application as needed.
#   warnings.warn(
# /mnt/cache/herunze/miniconda3/envs/plant/lib/python3.8/site-packages/mmdet/utils/setup_env.py:48: UserWarning: Setting MKL_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being 
# overloaded, please further tune the variable for optimal performance in your application as needed.

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p shlab-perceptionx \
    --mpi=pmi2 \
    --job-name=${job_name} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python tools/train.py ${CONFIG} ${PY_ARGS} 2>&1|tee log/train-$now.log &
    # -w SH-IDC1-10-140-0-227 \
    # --quotatype=spot \