#!/bin/bash
cd `dirname $0`
set -x -e -o pipefail -u

n_list=`seq 10000 10000 100000`
s=10

cmake -DCMAKE_CXX_FLAGS="-O3 -DENABLE_PRINT -DGRAVITY_CHECK_TIME" ..
make

#rm -i gravity_time.csv
for n in ${n_list[@]}; do
    ./nbody_benchmark_gravity $n $s 2>> gravity_time_double.csv
done

cmake -DCMAKE_CXX_FLAGS="-O3 -DENABLE_PRINT -DGRAVITY_CHECK_TIME -DGRAVITY_FLOAT" ..
make

#rm -i gravity_time.csv
for n in ${n_list[@]}; do
    ./nbody_benchmark_gravity $n $s 2>> gravity_time_float.csv
done
