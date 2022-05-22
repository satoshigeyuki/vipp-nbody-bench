import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.figure(figsize=(6, 6))

for p in [2, 4, 6, 8]:
    df = pd.read_csv(f"gravity_p={p}.csv", names=["delta_t", "error"])
    plt.plot(df["delta_t"], df["error"], label=f"p={p}")

guide_range = np.array([1e-3, 1])
plt.plot(guide_range, guide_range ** 5, label="delta_t^5", linestyle="dotted")

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
plt.xlabel("delta_t")
plt.ylabel("error")
plt.xscale("log")
plt.yscale("log")
plt.savefig("gravity_p.pdf", bbox_inches="tight")

