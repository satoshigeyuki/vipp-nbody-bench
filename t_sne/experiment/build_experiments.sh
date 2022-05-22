#!/bin/sh

CLANG=${CLANG:-clang++}
$CLANG --version > clang_version

for theta in 0.3 0.4 0.5
do
    cmake -DCMAKE_CXX_COMPILER=$CLANG -DCMAKE_CXX_FLAGS="-Ofast -DTSNE_STDERR_CSV -DTSNE_FLOAT -DTSNE_THETA=$theta" ..
    make
    mv nbody_benchmark_t_sne t-sne_theta${theta}
done
