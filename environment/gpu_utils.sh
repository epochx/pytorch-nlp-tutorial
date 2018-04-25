#!/usr/bin/env bash

# copy this into your ~/.bashrc

# CUDA stuff
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$CUDA_HOME/lib

#GPU Usage Tools

gpustat() {
    watch --color -n 1 "python ~/gpustat.py -u -c 2> /dev/null"
}

set-visible-gpus() {
    GPU_ID=""
    for i in "$@"
    do
        GPU_ID="$GPU_ID $i"
    done
    export CUDA_VISIBLE_DEVICES=$GPU_ID
}

reset-visible-gpus() {
    unset CUDA_VISIBLE_DEVICES
}

get-total-gpus() {
    GPUS_LEN=$(($(nvidia-smi -L | wc -l) - 1))
    GPUS=""
    for i in $(seq 0 $GPUS_LEN);
    do
        GPUS="$GPUS ${i}"
    done
    echo $GPUS
}

get-visible-gpus() {
    if [ -n "$CUDA_VISIBLE_DEVICES" ];
    then
        echo $CUDA_VISIBLE_DEVICES;
    else
        get-total-gpus;
    fi
