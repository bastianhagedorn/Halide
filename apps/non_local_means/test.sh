#!/usr/bin/env bash
PROCESS="."
if [[ $1 == "ref" ]]; then
    PROCESS="./filter_dillon" # fastest
else
    PROCESS="./filter_$1"
fi
OMP_NUM_THREADS=$2 HL_NUM_THREADS=$2 ${PROCESS}  ../images/rgb_small_noisy.png out.png 0.12
