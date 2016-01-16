#!/bin/bash
apps=`cat apps.txt`
#apps="blur"

halide_dir=${HOME}/ravi

start_time=`date "+%Y-%m-%d.%H%M"`
rundir="${HOSTNAME}-schmoo-${start_time}"
mkdir -p $rundir
cd $halide_dir
dirty=$((`git status --porcelain | grep ' M ' | wc -l` > 1))
commit=`git rev-parse head`
cd -
echo "hostname: ${HOSTNAME}," > $rundir/config.txt
echo "commit: ${commit}," >> $rundir/config.txt
echo "dirty: ${dirty}," >> $rundir/config.txt
echo "start_time: ${start_time}," >> $rundir/config.txt
exit



echo "This will remove {gen,run}.*.txt - OK?"
read
for app in $apps; do
    echo "============================================================" >> $errlog
    date "+$app (%Y-%m-%d %H:%M:%S)" >> $errlog
    echo ""
    echo "============================================================"
    pushd $app
    echo "$app"
    # make clean
    rm -f "gen.*.*.*.txt" "run.*.*.*.txt"
    for vec in 8; do
        echo "  vec: $vec"
        for par in 8; do
            echo "    par: $par"
            # 16kB  = 131072
            # 32kB  = 262144
            # 64kB  = 524288
            # 128kB = 1048576
            # 256kB = 2097152
            # 512kB = 4194304
            # 1MB   = 8388608
            # 2MB   = 16777216
            # 4MB   = 33554432
            # 8MB   = 67108864
            for memsize in 131072 524288 2097152 8388608 33554432; do
            #for memsize in 131072 2097152; do
                echo "      memsize: $memsize"
                for balance in 1 4 8 12 16 20; do
                #for balance in 1 8 20; do
                    echo "        balance: $balance"
                    genfile="gen.$par.$vec.$balance.$memsize.txt"
                    export HL_AUTO_PARALLELISM=$par
                    export HL_AUTO_VEC_LEN=$vec
                    export HL_AUTO_BALANCE=$balance
                    export HL_AUTO_FAST_MEM_SIZE=$memsize
                    if [[ -f ./regen.sh ]]; then
                        bash ./regen.sh 2>> err.log > $genfile
                    else
                        rm -f "*.o"
                        make -s cleangen 2>> err.log
                        make -s gen > $genfile 2>> err.log
                    fi
                    for runthreads in 4; do
                        echo "          numthreads: $runthreads"
                        runfile=run.$par.$vec.$balance.$memsize.$runthreads.txt
                        cat $genfile > $runfile
                        ./test.sh auto $runthreads >> \
                            $runfile \
                            2> err.log
                    done

                done
            done
        done
    done
    popd
done
