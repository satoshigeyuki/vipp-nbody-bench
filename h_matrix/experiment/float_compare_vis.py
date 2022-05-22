import pandas as pd
import matplotlib.pyplot as plt
import sys

# 軸ラベルからグラフ出力する軸を選択する
LABELS = [
    "N",  # 底面の分割数
    "matrix_size",  # 行列サイズ
    "residual_error",  # 近似行列との乗算誤差
    "approximation_time(sec)",  # 近似行列の構築にかかった時間
    "multiply_time(sec)",  # 近似行列のベクトルとの乗算にかかった時間
]
assert len(sys.argv) == 3
x = sys.argv[1]
y = sys.argv[2]

df = pd.read_csv(f"h_matrix_double.csv", names=LABELS)
plt.plot(df[x], df[y], label="double", marker=".")

df = pd.read_csv(f"h_matrix_float.csv", names=LABELS)
plt.plot(df[x], df[y], label="float", linestyle="dashed", marker=".")

df = pd.read_csv(f"h_matrix_lowrank.csv", names=LABELS)
plt.plot(df[x], df[y], label="lowrank", linestyle="dotted", marker=".")

plt.legend()
plt.xlabel(x)
plt.ylabel(y)
#plt.xscale("log")
#plt.yscale("log")
plt.tight_layout()
plt.savefig(f"h_matrix-float-{x}-{y}.pdf")
