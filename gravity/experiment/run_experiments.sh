#!/bin/sh

t=15
epsilon=0

Ns="500 1000 2000 4000 8000 16000 32000 64000"

for N in $Ns
do
    p=4
    s=$((2*$p*$p))
    ./gravity_delta-t_double $N $s $epsilon  > log_gravity_delta-t_double_p${p}_N${N}_s${s} 2> gravity_delta-t_double_N${N}.csv
    ./gravity_delta-t_t-double $N $s $epsilon > log_gravity_delta-t_t-double_p${p}_N${N}_s${s} 2> gravity_delta-t_t-double_N${N}.csv
    ./gravity_delta-t_float $N $s $epsilon  > log_gravity_delta-t_float_p${p}_N${N}_s${s}  2> gravity_delta-t_float_N${N}.csv
done

for p in 4 6 8
do
    rm -f gravity_time_double_p${p}.csv gravity_time_t-double_p${p}.csv gravity_time_float_p${p}.csv
done
for N in $Ns
do
    for p in 4 6 8
    do
        s=$((2*$p*$p))
        for _ in $(seq $t)
        do
            ./gravity_time_double_p$p $N $s $epsilon
        done > log_gravity_time_double_p${p}_N${N}_s${s} 2>> gravity_time_double_p${p}.csv

        for _ in $(seq $t)
        do
            ./gravity_time_t-double_p$p $N $s $epsilon
        done > log_gravity_time_t-double_p${p}_N${N}_s${s} 2>> gravity_time_t-double_p${p}.csv

        for _ in $(seq $t)
        do
            ./gravity_time_float_p$p $N $s $epsilon
        done > log_gravity_time_float_p${p}_N${N}_s${s} 2>> gravity_time_float_p${p}.csv
    done
done
