#!/bin/bash

label="golden"

qsub -N $label ${HOME}/ravi/apps/qsub/bench.pbs -F "-l $label -p'1 6'"
