#!/bin/sh

# arithmetic
for arith in double float low-rank
do
    for p in $(seq 3 6)
    do
        rm -f h-matrix_${arith}_p${p}.csv
        for n in $(seq 5 5 60)
        do
            ./h-matrix_${arith} $n $p > log_h-matrix_${arith}_p${p}_n${n} 2>> h-matrix_${arith}_p${p}.csv
        done
    done
done

# alpha
for alpha in 0.0 0.25 0.5 0.75 1.0 1.25 1.5 2.0 3.0 5.0
do
    rm -f h-matrix_alpha${alpha}.csv
    for n in 6 7 10 13 17 22 28 37 48
    do
        ./h-matrix_alpha${alpha} $n > log_h-matrix_alpha${alpha}_n${n} 2>> h-matrix_alpha${alpha}.csv
    done
done

# split
for split in naive until-n multiple-n
do
    rm -f h-matrix_split_${split}.csv
    for n in 6 7 10 13 17 22 28 37 48 63
    do
        ./h-matrix_split_${split} > log_h-matrix_split_${split}_n${n} $n 2>> h-matrix_split_${split}.csv
    done
done
