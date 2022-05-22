import pandas as pd
import matplotlib.pyplot as plt

experiment_names = ["naive", "until_n", "multiple_n"]

# 軸ラベルからグラフ出力する軸を選択する
LABELS = [
    "N",  # 底面の分割数
    "matrix_size",  # 行列サイズ
    "residual_error",  # 近似行列との乗算誤差
    "approximation_time",  # 近似行列の構築にかかった時間
    "multiply_time",  # 近似行列のベクトルとの乗算にかかった時間
]
x = "matrix_size"
y = "residual_error"

for n in experiment_names:
    df = pd.read_csv(f"h_matrix_split_{n}.csv", names=LABELS)
    plt.plot(df[x], df[y], label=n)

plt.legend()
plt.xlabel(x)
y = "multiply_time [ms]"
plt.ylabel(y)
plt.xscale("log")
plt.yscale("log")
plt.savefig(f"h_matrix_split-{x}-{y}.pdf")
