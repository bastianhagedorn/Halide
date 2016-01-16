#!/usr/bin/env bash
PROCESS="./process_$1"
OMP_NUM_THREADS=$2 HL_NUM_THREADS=$2 ${PROCESS} ../images/rgb.png 8 1 1 10 out.png
