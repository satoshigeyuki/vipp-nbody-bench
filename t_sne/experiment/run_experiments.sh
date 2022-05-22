#!/bin/sh

t=1000

rm -f t-sne.csv

for n in $(seq 3)
do
    for theta in 0.3 0.4 0.5
    do
        N=$(($n*70000))
        ./t-sne_theta$theta $n $t > log_t-sne_theta${theta}_N${N}_t${t} 2>> t-sne.csv
    done
done
