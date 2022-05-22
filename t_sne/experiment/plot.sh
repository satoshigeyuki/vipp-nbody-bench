#!/bin/bash
set -x
cd `dirname $0`
if [$# != 2]; then
    echo "usage: ./plot.sh OUT_BASENAME TITLE_NAME"
    exit 1
fi

# 描画する点の大きさ
psvalue=0.1

# 描画範囲のパディング
pad=2.5

# 出力ファイル名
OUT_BASENAME=$1

# タイトル
TITLE_NAME=$2

# 出力ファイルを文字ごとに分割
for i in {0..9}; do
    awk -F "," "(\$3==${i}){print}" ${OUT_BASENAME}.csv > "output${i}.csv"
done

# 描画範囲を設定
min_x=`echo $(cut -d, -f1 < ${OUT_BASENAME}.csv | sort -g | head -1) - $pad | bc`
max_x=`echo $(cut -d, -f1 < ${OUT_BASENAME}.csv | sort -g | tail -1) + $pad | bc`
min_y=`echo $(cut -d, -f2 < ${OUT_BASENAME}.csv | sort -g | head -1) - $pad | bc`
max_y=`echo $(cut -d, -f2 < ${OUT_BASENAME}.csv | sort -g | tail -1) + $pad | bc`

# 描画
gnuplot -e "minx='${min_x}';maxx='${max_x}';miny='${min_y}';maxy='${max_y}';psvalue='${psvalue}';psvalue2='1';output_file='${OUT_BASENAME}.pdf';title_name='${TITLE_NAME}'" plot.g
