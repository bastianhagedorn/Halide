#!/bin/bash
APPS=`cat apps.txt`
TESTS=`cat tests.txt`
CONV=`cat conv.txt`
BATCH=""
while [[ $# > 0 ]]
do
echo $arg
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
    -c|--conv)
    echo "Conv"
    BATCH+=" $CONV"
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
    make clean; make bench;
    # TODO: how to log gen data?
    cd ../;
done
#python benchmark.py
