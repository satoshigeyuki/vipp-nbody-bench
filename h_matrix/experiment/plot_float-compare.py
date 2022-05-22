import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import sys

# 軸ラベルからグラフ出力する軸を選択する
LABELS = [
    "n",  # 底面の分割数
    "N",  # 行列サイズ
    "Relative errors",  # 近似行列との乗算誤差
    "Approximation time (sec)",  # 近似行列の構築にかかった時間
    "Multiplication time (sec)",  # 近似行列のベクトルとの乗算にかかった時間
]
assert len(sys.argv) == 3
x = sys.argv[1]
y = sys.argv[2]

df = pd.read_csv(f"h-matrix_double.csv", names=LABELS)
plt.plot(df[x], df[y], label="double", marker=".")

df = pd.read_csv(f"h-matrix_float.csv", names=LABELS)
plt.plot(df[x], df[y], label="float", linestyle="dashed", marker=".")

df = pd.read_csv(f"h-matrix_low-rank.csv", names=LABELS)
plt.plot(df[x], df[y], label="low-rank", linestyle="dotted", marker=".")

plt.legend()
plt.xlabel(x)
plt.ylabel(y)
#plt.xscale("log")
#plt.yscale("log")
plt.tight_layout()
plt.savefig(f"h-matrix_float-compare_{x}-{y}.pdf")
