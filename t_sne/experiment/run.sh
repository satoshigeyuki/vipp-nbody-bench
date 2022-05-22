#!/bin/bash
cd `dirname $0`
set -x -e -o pipefail -u

n=${1:-1}
t=${2:-1000}
N=${3:-$((n*70000))}
BASENAME="tsne_output_ours_$N"
TITLE_NAME=ours\,N=$N,t=$t

if [ $# -eq 3 ]; then
    cmake -DCMAKE_CXX_FLAGS="-O3 -DENABLE_PRINT -DTSNE_INPUT_RESIZE=$N" ..
elif [ $# -le 2 ]; then
    cmake -DCMAKE_CXX_FLAGS="-O3 -DENABLE_PRINT" ..
else
    echo "usage: ./run.sh [AUGMENT_COEFF [NUM_ITERATIONS [INPUT_RESIZE]]]"
    exit 1
fi

make
./nbody_benchmark_t_sne $n $t 2> ${BASENAME}.csv

bash plot.sh $BASENAME $TITLE_NAME

