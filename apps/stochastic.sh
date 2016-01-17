#!/bin/bash
app=${1:-local_laplacian}
num_samples=${NUM_SAMPLES:-2}
threads=${THREADS:-6}

halide="${HOME}/auto-halide"

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

for (( i = 0; i < $num_samples; i++ )); do
    seed=`shuf -i 1-2000000000 -n 1`
    res="${outdir}/$app.$seed.res"
    echo "[$app.$seed]" >> $res
    echo "hostname: ${HOSTNAME}" >> $res
    echo "date: `date`" >> $res
    
    echo " "
    date "+%H:%M:%S"
    echo "Making $app $seed..."

    HL_AUTO_RANDOM_SEED=$seed make -s auto >> $res 2> $errlog

    echo "running..."

    numactl --cpunodebind 0 --membind 0 ./test.sh auto $threads >> $res 2> $errlog
done

make -s clean

echo "Done!"
