import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import product

x_label = "N"
y_label = "eval_duration"

df = pd.read_csv(f"gravity_time_double.csv", names=["N", "s", "alpha", "c", "eval_duration"])
plt.plot(df[x_label], df[y_label], marker=".", label="double")

df = pd.read_csv(f"gravity_time_float.csv", names=["N", "s", "alpha", "c", "eval_duration"])
plt.plot(df[x_label], df[y_label], marker=".", label="float")

plt.legend()
plt.xlim(0, 105000)
plt.ylim(0, 850)
#plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
plt.xlabel("number of points")
plt.ylabel("time [sec]")
#plt.xscale("log")
#plt.yscale("log")
plt.savefig("gravity_time.pdf", bbox_inches="tight")

