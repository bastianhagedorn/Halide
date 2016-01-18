#!/usr/bin/env bash
if [[ $1 == "ref" ]]; then
    sched=1
else
    sched=-1
fi
./conv_bench $sched
