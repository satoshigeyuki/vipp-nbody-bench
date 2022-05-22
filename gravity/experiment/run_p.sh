set -xeu
n=1000
s=10
epsilon=0
subdivide_ref_timestep=2

for p in 2 4 6 8; do
    cmake -DCMAKE_CXX_FLAGS="-O3 -DGRAVITY_P=$p -DGRAVITY_SEARCH_DELTA_T -DGRAVITY_SUBDIVIDE_REF_TIMESTEP=$subdivide_ref_timestep" ..
    make
    ./nbody_benchmark_gravity $n $s $epsilon 2> gravity_p=${p}.csv
done

python3 make_graph_p.py
