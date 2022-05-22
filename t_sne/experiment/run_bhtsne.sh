#!/bin/bash
cd `dirname $0`
set -e -o pipefail -u

if [ $# != 1 ]; then
    echo "usage: ./run_bhtsne.sh INPUT_SIZE"
    exit 1
fi

# 入力サイズ
N=${1:-70000}

# 出力ファイル名
BASENAME="tsne_output_bhtsne_$N"

# bhtsne を実行
python3 run_bhtsne.py $N $BASENAME

# グラフタイトル
TITLE_NAME=bhtsne\,N=${N}\,t=1000

# 出力を記録
bash plot.sh $BASENAME $TITLE_NAME
