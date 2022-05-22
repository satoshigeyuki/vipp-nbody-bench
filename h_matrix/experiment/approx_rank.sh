n=${1:-20}
cmake_flag_base="-O3 -DH_MATRIX_PRINT_APPROX_MATRIX"
cmake -DCMAKE_CXX_FLAGS="$cmake_flag_base" ..
make
./nbody_benchmark_h_matrix $n 2> approx_matrix_double_${n}.svg

cmake -DCMAKE_CXX_FLAGS="$cmake_flag_base -DH_MATRIX_FLOAT" ..
make
./nbody_benchmark_h_matrix $n 2> approx_matrix_float_${n}.svg

cmake -DCMAKE_CXX_FLAGS="$cmake_flag_base -DH_MATRIX_FLOAT -DH_MATRIX_SAME_RANK_AS" ..
make
./nbody_benchmark_h_matrix $n 2> approx_matrix_lowrank_${n}.svg
