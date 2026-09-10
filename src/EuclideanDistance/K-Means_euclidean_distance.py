from sklearn.cluster import KMeans
import numpy as np

# Simple data
X = np.array([
    [1, 2],
    [1, 4],
    [1, 0],
    [10, 2],
    [10, 4],
    [10, 0]
])

# K-Means com distância euclidiana (default)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

print("Centroides:")
print(kmeans.cluster_centers_)

print("Labels:")
print(kmeans.labels_)
