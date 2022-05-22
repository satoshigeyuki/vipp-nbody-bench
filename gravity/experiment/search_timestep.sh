#!/bin/bash
cd `dirname $0`
set -x -e -o pipefail -u

n_list=(500 1000 2000 4000 8000 16000 32000 64000)
s=10
epsilon=0
subdivide_ref_timestep=2

cmake -DCMAKE_CXX_FLAGS="-O3 -DGRAVITY_SEARCH_DELTA_T -DGRAVITY_SUBDIVIDE_REF_TIMESTEP=$subdivide_ref_timestep" ..
make

for n in ${n_list[@]}; do
    ./nbody_benchmark_gravity $n $s $epsilon 2> gravity_double_${n}.csv
done

cmake -DCMAKE_CXX_FLAGS="-O3 -DGRAVITY_SEARCH_DELTA_T -DGRAVITY_SUBDIVIDE_REF_TIMESTEP=$subdivide_ref_timestep -DGRAVITY_FLOAT" ..
make

for n in ${n_list[@]}; do
    ./nbody_benchmark_gravity $n $s $epsilon 2> gravity_float_${n}.csv
done

python3 make_graph_delta_t.py
python3 make_graph_delta_t_double.py
