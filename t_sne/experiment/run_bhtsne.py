from bhtsne import tsne
import sys
import os
from sklearn.datasets import fetch_openml

print("read input...")
assert len(sys.argv) == 3, "usage: python run_bhtsne.py SUBSET_SIZE_MNIST OUT_BASENAME"
SUBSET_SIZE_MNIST, OUT_BASENAME = int(sys.argv[1]), str(sys.argv[2])
assert 1 <= SUBSET_SIZE_MNIST <= 70000, SUBSET_SIZE_MNIST
assert not os.path.exists(OUT_BASENAME + ".csv"), OUT_BASENAME

print("fetch MNIST data...")
data, label = fetch_openml('mnist_784', version=1, return_X_y=True)

print("run bhtsne...")
points = tsne(data[:SUBSET_SIZE_MNIST])

print("dump output...")
with open(OUT_BASENAME + ".csv", "w") as f:
    f.write("\n".join([f"{x}, {y}, {i}" for (x, y), i in zip(points, label)])) 
