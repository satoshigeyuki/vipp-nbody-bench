#!/bin/sh

CLANG=${CLANG:-clang++}
$CLANG --version > clang_version

cmake_flag_base="-Ofast -DENABLE_PRINT_ERROR -DH_MATRIX_MULTIPLY_REPEAT=1000"

# double
cmake -DCMAKE_CXX_COMPILER=$CLANG -DCMAKE_CXX_FLAGS="$cmake_flag_base -DH_MATRIX_APPROX_ALPHA=1" ..
make
mv nbody_benchmark_h_matrix h-matrix_double

# float
cmake -DCMAKE_CXX_COMPILER=$CLANG -DCMAKE_CXX_FLAGS="$cmake_flag_base -DH_MATRIX_APPROX_ALPHA=1 -DH_MATRIX_FLOAT" ..
make
mv nbody_benchmark_h_matrix h-matrix_float

# low-rank float
cmake -DCMAKE_CXX_COMPILER=$CLANG -DCMAKE_CXX_FLAGS="$cmake_flag_base -DH_MATRIX_APPROX_ALPHA=1 -DH_MATRIX_FLOAT -DH_MATRIX_SAME_RANK_AS" ..
make
mv nbody_benchmark_h_matrix h-matrix_low-rank

# alpha comparison
for alpha in 0.0 0.25 0.5 0.75 1.0 1.25 1.5 2.0 3.0 5.0
do
    cmake -DCMAKE_CXX_COMPILER=$CLANG -DCMAKE_CXX_FLAGS="$cmake_flag_base -DH_MATRIX_APPROX_ALPHA=$alpha" ..
    make
    mv nbody_benchmark_h_matrix h-matrix_alpha${alpha}
done

# split comparison
cmake -DCMAKE_CXX_COMPILER=$CLANG -DCMAKE_CXX_FLAGS="$cmake_flag_base -DH_MATRIX_SPLIT_THRESHOLD=1 -DH_MATRIX_ROUNDDOWN_BASE=1" ..
make
mv nbody_benchmark_h_matrix h-matrix_split_naive

cmake -DCMAKE_CXX_COMPILER=$CLANG -DCMAKE_CXX_FLAGS="$cmake_flag_base -DH_MATRIX_ROUNDDOWN_BASE=1" ..
make
mv nbody_benchmark_h_matrix h-matrix_split_until-n

cmake -DCMAKE_CXX_COMPILER=$CLANG -DCMAKE_CXX_FLAGS="$cmake_flag_base" ..
make
mv nbody_benchmark_h_matrix h-matrix_split_multiple-n
