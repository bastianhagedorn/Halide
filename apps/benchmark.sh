#!/bin/bash
APPS=`cat apps.txt`
RACES=`cat raceapps.txt`
TESTS=`cat tests.txt`
CONV=`cat conv.txt`

rand=`LC_CTYPE=C tr -dc a-f0-9 < /dev/urandom | fold -w 8 | head -n 1`

myrealpath () {
  [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

LABEL=""
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
    -l|--label)
    LABEL="-$2"
    shift # past argument
    ;;
    -p|--threads)
    export THREADS_TO_TEST="$2"
    shift # past argument
    ;;
    *)
            # unknown option
    ;;
esac
shift # past argument or value
done
if [[ $BATCH == "" ]]; then
    BATCH="$APPS $RACES"
fi

echo ${rand}

start_time=`date "+%Y-%m-%d.%H%M%S"`
rundirname="bench-${HOSTNAME}-${rand}${LABEL}"
mkdir -p $rundirname
rundir=`readlink -f $rundirname`

echo "Running in $rundirname"

halide_hash=`openssl md5 ../lib/libHalide.a  | sed 's/^.* //'`

echo "hostname: ${HOSTNAME}" >> $rundir/config.txt
echo "start_time: ${start_time}" >> $rundir/config.txt
echo "libHalide.a: ${halide_hash}" >> $rundir/config.txt
echo "batch: ${BATCH}" >> $rundir/config.txt

for app in $BATCH; do
    cd $app;
    echo "============================================================"
    echo "                  BENCHMARKING $app"
    echo "============================================================"
    mkdir -p "${rundir}/${app}"

    make -s clean;
    make -s auto > "${rundir}/${app}/gen.txt" 2>> "${rundir}/${app}/err.log"
    make -s bench;
    echo ${rundir}/${app}
    cat *_perf.txt
    mv *_perf.txt "${rundir}/${app}"
    cd ../;
done

end_time=`date "+%Y-%m-%d.%H%M%S"`

echo "end_time: ${end_time}" >> $rundir/config.txt
echo "Finished: ${end_time}" >> "$rundir/done.txt"
