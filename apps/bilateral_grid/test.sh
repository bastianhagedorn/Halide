#!/usr/bin/env bash
PROCESS="./filter_$1"
OMP_NUM_THREADS=$2 HL_NUM_THREADS=$2 ${PROCESS} ../images/gray.png out.png 0.1 10
