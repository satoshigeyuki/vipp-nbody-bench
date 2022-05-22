import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import numpy as np

# 実験に使用したalpha
alphas = ["0.0", "0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "2.0", "3.0", "5.0"]

# 軸ラベルからグラフ出力する軸を選択する
LABELS = [
    "N",  # 底面の分割数
    "matrix_size",  # 行列サイズ
    "residual_error",  # 近似行列との乗算誤差
    "approximation_time",  # 近似行列の構築にかかった時間
    "multiply_time",  # 近似行列のベクトルとの乗算にかかった時間
]
x = "N"
x2 = "matrix_size"
y = "multiply_time"

data_y = []
for alpha in alphas:
    df = pd.read_csv(f"h_matrix_approx_alpha_{alpha}.csv", names=LABELS)
    data_y.append(df[y])
data_y = np.array(data_y).transpose()

for s, s2, d in zip(df[x], df[x2], data_y):
    plt.plot([float(alpha) for alpha in alphas], d, label=f"N={s} (size: {s2}x{s2})")

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
plt.xlabel("alpha")
plt.ylabel("multiply_time [ms]")
#plt.xscale("log")
#plt.yscale("log")
plt.savefig(f"h_matrix-alpha-{x}-{y}.pdf", bbox_inches="tight")
