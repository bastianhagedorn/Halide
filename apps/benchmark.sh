#!/bin/bash
APPS=`cat apps.txt`
TESTS=`cat tests.txt`
for app in $APPS $TESTS; do
    cd $app;
    echo "============================================================"
    echo "                  BENCHMARKING $app"
    echo "============================================================"
    make clean; make bench;
    cd ../;
done
#python benchmark.py
