#!/bin/bash
APPS=`cat apps.txt`
TESTS=`cat tests.txt`
BATCH=""
while [[ $# > 0 ]]
do
echo "arg"
key="$1"
case $key in
    -a|--apps)
    echo "Apps"
    BATCH+=" $APPS"
    ;;
    -t|--tests)
    echo "Tests"
    BATCH+=" $TESTS"
    ;;
    -e|--extra)
    BATCH+=" $2"
    shift # past argument
    ;;
    *)
            # unknown option
    ;;
esac
shift # past argument or value
done
if [[ $BATCH == "" ]]; then
    BATCH=$APPS
fi
for app in $BATCH; do
    cd $app;
    echo "============================================================"
    echo "                  BENCHMARKING $app"
    echo "============================================================"
    #make clean; make bench;
    cd ../;
done
#python benchmark.py
