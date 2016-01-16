#!/usr/bin/env bash
PROCESS="./test_$1"
OMP_NUM_THREADS=$2 HL_NUM_THREADS=$2 ${PROCESS}
