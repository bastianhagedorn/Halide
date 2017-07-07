#!/bin/bash
make clean

export HL_DEBUG_CODEGEN=0
#export HL_GPU_DEVICE=1
#export HL_JIT_TARGET=0
#export HL_NUMTHREADS=0
#export HL_OCL_DEVICE=0
#export HL_OCL_DEVICE_TYPE=0
#export HL_OCL_PLATFORM=0
#export HL_OCL_PLATFORM_NAME=0
#export HL_PROFILE=0
#export HL_TARGET=0
#export HL_TRACE=0
#export HL_TRACE_FILE=0

LLVM_CONFIG=/home/bastian/tools/llvm3.7/build/bin/llvm-config CLANG=/home/bastian/tools/llvm3.7/build/bin/clang make auto
LD_PRELOAD=/home/bastian/tools/heliumpp/libOpenCL.so ./test_auto_gpu
LD_PRELOAD=/home/bastian/tools/heliumpp/libOpenCL.so ./test_naive_gpu
