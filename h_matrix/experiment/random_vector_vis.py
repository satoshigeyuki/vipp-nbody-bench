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

df = pd.read_csv(f"h_matrix_standard_normal.csv", names=LABELS)
plt.plot(df[x], df[y], label="standard normal", marker=".")

df = pd.read_csv(f"h_matrix_standard_uniform.csv", names=LABELS)
plt.plot(df[x], df[y], label="standard uniform", linestyle="dashed", marker=".")

df = pd.read_csv(f"h_matrix_uniform_0.9to1.1.csv", names=LABELS)
plt.plot(df[x], df[y], label="uniform [0.9, 1.1]", linestyle="dashed", marker=".")

df = pd.read_csv(f"h_matrix_uniform_1to2.csv", names=LABELS)
plt.plot(df[x], df[y], label="uniform [1, 2]", linestyle="dashed", marker=".")

plt.legend()
plt.xlabel(x)
plt.ylabel(y)
#plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
plt.savefig(f"h_matrix-random-{x}-{y}.pdf")
