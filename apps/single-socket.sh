#!/bin/bash
if [[ -x `which numactl` ]]; then
    numactl --cpunodebind 0 --membind 0 $@
else
    echo "No numactl, running as is..."
    $@
fi
