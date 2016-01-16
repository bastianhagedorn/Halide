#!/bin/bash -x
set -e
set -o pipefail

src=${HOME}/ravi
build=${HOME}/auto-halide
#: ${LLVM_INCLUDE:?"LLVM_INCLUDE must be specified"}

cd $build

make -f ${src}/Makefile
rsync -avu --exclude "*bench_*" --exclude "*perf.txt" ${src}/apps/ ${build}/apps/
rsync -avu ${src}/tools/ ${build}/tools/
