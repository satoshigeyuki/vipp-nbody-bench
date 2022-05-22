#!/bin/bash
cd `dirname $0`
set -x -e -o pipefail -u

n=${1:-1}
t=${2:-1000}
N=${3:-$((n*70000))}
TITLE_NAME=ours\,N=$N,t=$t


if [ $# -eq 3 ]; then
    CMAKE_CXX_FLAGS_BASE="-O3 -DTSNE_FLOAT -DTSNE_INPUT_RESIZE=$N"
elif [ $# -le 2 ]; then
    CMAKE_CXX_FLAGS_BASE="-O3 -DTSNE_FLOAT"
else
    echo "usage: ./run_theta.sh [AUGMENT_COEFF [NUM_ITERATIONS [INPUT_RESIZE]]]"
    exit 1
fi

for theta in 0.1 0.2 0.3 0.4 0.5; do

    # evaluated solver = float, reference solver = double
    BASENAME="tsne_output_${N}_theta_${theta}"
    cmake -DCMAKE_CXX_FLAGS="$CMAKE_CXX_FLAGS_BASE -DTSNE_THETA=$theta" ..
    make
    ./nbody_benchmark_t_sne $n $t | tee ${BASENAME}.txt

done
