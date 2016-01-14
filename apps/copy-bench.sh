#!/bin/bash

REMOTE_PATH=$2
HOST=$1
rsync -amv --include='*_perf.txt' --include='*/' --exclude='*' ${HOST}:${REMOTE_PATH}/ ./
