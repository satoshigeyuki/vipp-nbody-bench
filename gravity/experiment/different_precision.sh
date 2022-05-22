n=${1:-10000}
s=${2:-10}
cmake -DCMAKE_CXX_FLAGS="-O3 -DENABLE_DIFFERENT_PRECISION" ..
make
./nbody_benchmark_gravity $n $s
