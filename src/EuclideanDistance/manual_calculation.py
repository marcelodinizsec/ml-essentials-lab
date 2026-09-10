import numpy as np
from sklearn.metrics import pairwise_distances

# Two points in space
A = np.array([[1, 2]])
B = np.array([[4, 6]])

# Manual Euclidean distance
dist_manual = np.sqrt(np.sum((A - B)**2))
print("Manual distance:", dist_manual)

# Using sklearn
dist_sklearn = pairwise_distances(A, B, metric='euclidean')[0][0]
print("sklearn distance:", dist_sklearn)