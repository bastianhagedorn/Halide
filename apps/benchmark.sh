#!/bin/bash
for app in cost_function_test overlap_test split_test tile_vs_inline_test \
           data_dependent_test parallel_test \
           blur local_laplacian interpolate bilateral_grid camera_pipe; do
    cd $app;
    make clean; make bench_ref; make bench_auto;
    cd ../;
done
python benchmark.py
