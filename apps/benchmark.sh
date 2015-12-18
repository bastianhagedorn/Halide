#!/bin/bash
for app in blur local_laplacian interpolate bilateral_grid camera_pipe; do
    cd $app;
    make clean; make bench_ref;
    cd ../;
done
python benchmark.py
