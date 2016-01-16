#!/bin/bash

REMOTE_PATH=$2
HOST=$1
rsync -amv --include='*_perf.txt' --include 'run.*.txt' --include 'gen.*.txt' --include='*/' --exclude='*' ${HOST}:${REMOTE_PATH}/ ./
