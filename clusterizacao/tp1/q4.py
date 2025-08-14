import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 1) Criar dataset
X, _ = make_blobs(n_samples=300, centers=5, cluster_std=0.6, random_state=42)

# 2) Aplicar K-Means para quantização
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# 3) "Reconstruir" pontos usando apenas o centróide do cluster
X_quantizado = centroids[labels]

# 4) Visualização
plt.figure(figsize=(10,4))

# Original
plt.subplot(1,2,1)
plt.scatter(X[:,0], X[:,1], c='gray', s=30, alpha=0.6)
plt.title("Dados Originais")

# Quantizado
plt.subplot(1,2,2)
plt.scatter(X_quantizado[:,0], X_quantizado[:,1], c=labels, cmap="tab10", s=30)
plt.scatter(centroids[:,0], centroids[:,1], c='red', marker='X', s=200, label='Centróides')
plt.title("Dados Quantizados")
plt.legend()

plt.show()
