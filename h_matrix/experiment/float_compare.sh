#!/bin/bash
cd `dirname $0`
set -x -e -o pipefail -u

n_list=`seq 5 5 40`
cmake_flag_base="-O3 -DENABLE_PRINT_ERROR -DH_MATRIX_MULTIPLY_REPEAT=1000"
#rm h_matrix_*.csv

# double
cmake -DCMAKE_CXX_FLAGS="$cmake_flag_base" ..
make
for n in ${n_list[@]}; do
    ./nbody_benchmark_h_matrix $n 2>> h_matrix_double.csv
done

# float
cmake -DCMAKE_CXX_FLAGS="$cmake_flag_base -DH_MATRIX_FLOAT" ..
make
for n in ${n_list[@]}; do
    ./nbody_benchmark_h_matrix $n 2>> h_matrix_float.csv
done

# float
cmake -DCMAKE_CXX_FLAGS="$cmake_flag_base -DH_MATRIX_FLOAT -DH_MATRIX_SAME_RANK_AS" ..
make
for n in ${n_list[@]}; do
    ./nbody_benchmark_h_matrix $n 2>> h_matrix_lowrank.csv
done

python3 float_compare_vis.py matrix_size residual_error
python3 float_compare_vis.py N residual_error
python3 float_compare_vis.py N "approximation_time(sec)"
python3 float_compare_vis.py N "multiply_time(sec)"
python3 float_compare_vis.py matrix_size "multiply_time(sec)"
