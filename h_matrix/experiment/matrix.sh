n=${1:-6}
cmake -DCMAKE_CXX_FLAGS="-O3 -DENABLE_PRINT_MATRIX" ..
make
./nbody_benchmark_h_matrix $n 2> output.csv
min=$(cut -d,  -f3 < output.csv | sort -g | head -1)
max=$(cut -d,  -f3 < output.csv | sort -g | tail -1)
echo "min='${min}';max='${max}';"
gnuplot -e "min_value='${min}';max_value='${max}'" matrix.g
