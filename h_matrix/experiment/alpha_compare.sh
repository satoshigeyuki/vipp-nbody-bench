#!/bin/bash
cd `dirname $0`
set -x -e -o pipefail -u

n_list=(6 7 10 13 17 22 28 37 48)
alpha_list=(0.0 0.25 0.5 0.75 1.0 1.25 1.5 2.0 3.0 5.0)
cmake_flag_base="-O3 -DENABLE_PRINT_ERROR -DH_MATRIX_MULTIPLY_REPEAT=1000"
#rm h_matrix_*.csv

for alpha in ${alpha_list[@]}; do
    cmake -DCMAKE_CXX_FLAGS="$cmake_flag_base -DH_MATRIX_APPROX_ALPHA=$alpha" ..
    make
    for n in ${n_list[@]}; do
        ./nbody_benchmark_h_matrix $n 2>> h_matrix_approx_alpha_$alpha.csv
    done
done

python3 alpha_compare_vis.py
