#!/bin/bash
app=${1:-lens_blur}
num_samples=${NUM_SAMPLES:-300}
threads=${THREADS:-6}

echo "Threads: $threads"
echo "Samples: $num_samples"

halide=`readlink -f ../`

function datestamp {
    date "+%Y-%m-%d.%H%M%S"
}

rundir=${halide}/apps/.${HOSTNAME}-$app
mkdir -p $rundir
rsync -a --exclude stochastic ${halide}/apps/$app/ $rundir/

outdir=${halide}/apps/$app/stochastic/
mkdir -p $outdir

errlog="${outdir}/${HOSTNAME}.err"

cd $rundir
echo $rundir
function test_seed {
    seed=$1
    
    res="${outdir}/$app.$seed.res"
    
    if [[ -f $res ]]; then
        echo "Default auto:"
        if grep runtime $res ; then
            echo "Skipping already-existing $seed"
            return
        fi
    fi
    echo "[$app.$seed]" >> $res
    echo "hostname: ${HOSTNAME}" >> $res
    echo "date: `date`" >> $res
    
    echo " "
    date "+%H:%M:%S"
    echo "Making $app $seed..."

    HL_AUTO_RANDOM_SEED=$seed make -s auto >> $res 2> $errlog

    echo "running..."

    numactl --cpunodebind 0 --membind 0 ./test.sh auto $threads >> $res 2> $errlog
    
    grep runtime $res
}

# Make sure our auto baseline is there
test_seed 0

#
# Do the actual run
#
for (( i = 0; i < $num_samples; i++ )); do
    seed=`shuf -i 1-2000000000 -n 1`
    test_seed $seed
done

make -s clean

echo "Done!"
