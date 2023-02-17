#!/usr/bin/env bash

set -x
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

path=$1
GPUS=$2
GPUS_PER_NODE=$2
name=$3
CPUS_PER_TASK=8


srun -p shlab-perceptionx \
    --mpi=pmi2 \
    --job-name=${name} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    bash ${path} 2>&1|tee log/train-$now.log &


# srun -p shlab-perceptionx --mpi=pmi2 --gres=gpu:0 --ntasks=0 --ntasks-per-node=1- --cpus-per-task=10 --kill-on-bad-exit=1 