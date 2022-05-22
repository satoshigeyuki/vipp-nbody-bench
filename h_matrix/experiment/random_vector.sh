#!/bin/bash
cd `dirname $0`
set -x -e -o pipefail -u

n_list=`seq 5 5 40`
cmake_flag_base="-O3 -DENABLE_PRINT_ERROR -DH_MATRIX_MULTIPLY_REPEAT=1000"

# standard normal distribution
cmake -DCMAKE_CXX_FLAGS="$cmake_flag_base" ..
make
for n in ${n_list[@]}; do
    ./nbody_benchmark_h_matrix $n 2>> h_matrix_standard_normal.csv
done

# standard uniform distribution
cmake -DCMAKE_CXX_FLAGS="$cmake_flag_base -DH_MATRIX_UNIFORM_DISTRIBUTION" ..
make
for n in ${n_list[@]}; do
    ./nbody_benchmark_h_matrix $n 2>> h_matrix_standard_uniform.csv
done

# uniform distribution, range = [0.9, 1.1]
cmake -DCMAKE_CXX_FLAGS="$cmake_flag_base -DH_MATRIX_UNIFORM_DISTRIBUTION -DH_MATRIX_DISTRIBUTION_A=0.9 -DH_MATRIX_DISTRIBUTION_B=1.1" ..
make
for n in ${n_list[@]}; do
    ./nbody_benchmark_h_matrix $n 2>> h_matrix_uniform_0.9to1.1.csv
done

# uniform distribution, range = [1, 2]
cmake -DCMAKE_CXX_FLAGS="$cmake_flag_base -DH_MATRIX_UNIFORM_DISTRIBUTION -DH_MATRIX_DISTRIBUTION_A=1 -DH_MATRIX_DISTRIBUTION_B=2" ..
make
for n in ${n_list[@]}; do
    ./nbody_benchmark_h_matrix $n 2>> h_matrix_uniform_1to2.csv
done

python3 random_vector_vis.py matrix_size residual_error
python3 random_vector_vis.py N residual_error
python3 random_vector_vis.py N "approximation_time(sec)"
python3 random_vector_vis.py N "multiply_time(sec)"
python3 random_vector_vis.py matrix_size "multiply_time(sec)"
