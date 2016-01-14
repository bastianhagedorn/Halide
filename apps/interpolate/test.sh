#!/usr/bin/env bash
if [[ $1 == "ref" ]]; then
    sched=""
else
    sched=-1
fi
./interpolate ../images/rgba.png out.png $sched
