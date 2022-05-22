import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import product

precisions = ["double"]
numbers = 500 * 2 ** np.arange(0, 8)
colors = [plt.cm.gist_rainbow(i*32) for i in range(8)]

plt.figure(figsize=(6, 6))

for p, (n, c) in product(precisions, zip(numbers, colors)):
    df = pd.read_csv(f"gravity_{p}_{n}.csv", names=["delta_t", "error"])
    plt.plot(df["delta_t"], df["error"], label=f"n={n}", color=c, linestyle=None if p == "double" else "dashed")

guide_range = np.array([1e-3, 1])
plt.plot(guide_range, guide_range ** 5, label="delta_t^5", linestyle="dotted")

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
plt.xlabel("delta_t")
plt.ylabel("error")
plt.xscale("log")
plt.yscale("log")
plt.savefig("gravity_delta_t_double.pdf", bbox_inches="tight")

