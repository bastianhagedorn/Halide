#!/usr/bin/env bash
if [[ $1 == "ref" ]]; then
    sched=0
else
    if [[ $1 == "naive" ]]; then
        export HL_AUTO_NAIVE=1
    fi
    sched=-1
fi
OMP_NUM_THREADS=$2 HL_NUM_THREADS=$2 ./max_filter ../images/rgb.png out.png $sched
