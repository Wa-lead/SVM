import numpy as np
from sklearn import datasets

X, y = datasets.make_blobs(
    n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
)
y = np.where(y == 0, -1, 1)


w = np.array([0,0])

print(X.T @ w)