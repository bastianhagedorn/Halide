#!/usr/bin/env bash
if [[ $1 == "ref" ]]; then
    sched=0
else
    sched=-1
fi
./mat_mul $sched
