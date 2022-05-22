#!/bin/bash
cd `dirname $0`
set -x -e -o pipefail -u

n_list=(6 7 10 13 17 22 28 37 48 63)
cmake_flag_base="-O3 -DENABLE_PRINT_ERROR -DH_MATRIX_MULTIPLY_REPEAT=1000"
#rm h_matrix_*.csv

cmake -DCMAKE_CXX_FLAGS="$cmake_flag_base -DH_MATRIX_SPLIT_THRESHOLD=1 -DH_MATRIX_ROUNDDOWN_BASE=1" ..
make
for n in ${n_list[@]}; do
    ./nbody_benchmark_h_matrix $n 2>> h_matrix_split_naive.csv
done

cmake -DCMAKE_CXX_FLAGS="$cmake_flag_base -DH_MATRIX_ROUNDDOWN_BASE=1" ..
make
for n in ${n_list[@]}; do
    ./nbody_benchmark_h_matrix $n 2>> h_matrix_split_until_n.csv
done

cmake -DCMAKE_CXX_FLAGS="$cmake_flag_base" ..
make
for n in ${n_list[@]}; do
    ./nbody_benchmark_h_matrix $n 2>> h_matrix_split_multiple_n.csv
done

python3 split_compare_vis.py
