n=${1:-60}

# 特異値の総和が全体の threshold 倍に達したところで近似する.
# 精度は行列成分と同じ型になる（デフォルトは double ）
thresholds=(0.9 0.999 0.99999 1)
cp error-template.g error.g

threshold_count=0
for threshold in ${thresholds[@]}; do
    rm -f "output${threshold}.csv"
    cmake -DCMAKE_CXX_FLAGS="-O3 -DENABLE_PRINT_ERROR -DH_MATRIX_RANK_THRESHOLD=${threshold}" ..
    make
    for i in `seq 6 ${n}`; do
        echo ${i}
        ./nbody_benchmark_h_matrix ${i} >/dev/null 2>>"output${threshold}.csv"
    done
    let threshold_count++
    if [ $threshold_count -eq ${#thresholds[@]} ]; then
        echo "\"output${threshold}.csv\" using 1:3 title \"threshold = ${threshold}\" with lines" >> error.g
    else
        echo "\"output${threshold}.csv\" using 1:3 title \"threshold = ${threshold}\" with lines,\\" >> error.g
    fi
done
gnuplot -e "maxN='${n}'" error.g
