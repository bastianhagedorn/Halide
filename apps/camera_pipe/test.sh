#!/usr/bin/env bash
PROCESS="./process_$1"
OMP_NUM_THREADS=$2 HL_NUM_THREADS=$2 ${PROCESS} ../images/bayer_raw.png 3700 2.0 50 10 out.png
