#!/bin/sh

CLANG=${CLANG:-clang++}
$CLANG --version > clang_version

common_opt="-Ofast"
subdivide_ref_timestep=2

cmake -DCMAKE_CXX_COMPILER=$CLANG -DCMAKE_CXX_FLAGS="$common_opt -DGRAVITY_SEARCH_DELTA_T -DGRAVITY_SUBDIVIDE_REF_TIMESTEP=$subdivide_ref_timestep" ..
make
mv nbody_benchmark_gravity gravity_delta-t_double

cmake -DCMAKE_CXX_COMPILER=$CLANG -DCMAKE_CXX_FLAGS="$common_opt -DGRAVITY_SEARCH_DELTA_T -DGRAVITY_SUBDIVIDE_REF_TIMESTEP=$subdivide_ref_timestep -DGRAVITY_TIME_DOUBLE" ..
make
mv nbody_benchmark_gravity gravity_delta-t_t-double

cmake -DCMAKE_CXX_COMPILER=$CLANG -DCMAKE_CXX_FLAGS="$common_opt -DGRAVITY_SEARCH_DELTA_T -DGRAVITY_SUBDIVIDE_REF_TIMESTEP=$subdivide_ref_timestep -DGRAVITY_FLOAT" ..
make
mv nbody_benchmark_gravity gravity_delta-t_float

for p in 4 6 8
do
    cmake -DCMAKE_CXX_COMPILER=$CLANG -DCMAKE_CXX_FLAGS="$common_opt -DGRAVITY_CHECK_TIME -DGRAVITY_P=$p" ..
    make
    mv nbody_benchmark_gravity gravity_time_double_p$p

    cmake -DCMAKE_CXX_COMPILER=$CLANG -DCMAKE_CXX_FLAGS="$common_opt -DGRAVITY_CHECK_TIME -DGRAVITY_P=$p -DGRAVITY_TIME_DOUBLE" ..
    make
    mv nbody_benchmark_gravity gravity_time_t-double_p$p

    cmake -DCMAKE_CXX_COMPILER=$CLANG -DCMAKE_CXX_FLAGS="$common_opt -DGRAVITY_CHECK_TIME -DGRAVITY_P=$p -DGRAVITY_FLOAT" ..
    make
    mv nbody_benchmark_gravity gravity_time_float_p$p
done
