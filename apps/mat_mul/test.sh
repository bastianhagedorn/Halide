#!/usr/bin/env bash
if [[ $1 == "ref" ]]; then
    sched=0
else
    if [[ $1 == "naive" ]]; then
        export HL_AUTO_NAIVE=1
    elif [[ $1 == "sweep" ]]; then
        export HL_AUTO_SWEEP=1 HL_AUTO_PARALLELISM=12 HL_AUTO_VEC_LEN=16 HL_AUTO_BALANCE=20 HL_AUTO_FAST_MEM_SIZE=32768
    elif [[ $1 == "rand" ]]; then
        export HL_AUTO_RAND=1 HL_AUTO_PARALLELISM=12 HL_AUTO_VEC_LEN=16 HL_AUTO_BALANCE=20 HL_AUTO_FAST_MEM_SIZE=32768
        #export HL_AUTO_RAND=1 HL_AUTO_PARALLELISM=8 HL_AUTO_VEC_LEN=8 HL_AUTO_BALANCE=10 HL_AUTO_FAST_MEM_SIZE=131072
    elif [[ $1 == "naive_gpu" ]]; then
        export HL_AUTO_NAIVE=1
    fi
    if [[ $1 == "naive_gpu" ]] || [[ $1 == "auto_gpu" ]]; then
        sched=-2
    else
        sched=-1
    fi
fi
OMP_NUM_THREADS=$2 HL_NUM_THREADS=$2 ./mat_mul $sched
