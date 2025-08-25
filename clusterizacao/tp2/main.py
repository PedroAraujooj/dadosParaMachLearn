import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform

np.random.seed(42)

# -----------------------------
# 7) Swiss roll + Hierarchical clustering (Ward)
# -----------------------------
n_samples_full = 1000
X, t = make_swiss_roll(n_samples=n_samples_full, noise=0.05, random_state=42)
X = StandardScaler().fit_transform(X)  #


Z_full = linkage(X, method="ward", metric="euclidean")

# Heuristic (Q11) to estimate k from relative jumps in merge distances
def choose_k_via_relative_jump(Z, tail=120):
    n = Z.shape[0] + 1
    d = Z[:, 2]
    m = len(d)
    tail = min(tail, m - 1)
    start = m - tail - 1
    start = max(0, start)
    d_tail = d[start:]
    rel = np.diff(d_tail) / np.maximum(d_tail[:-1], 1e-12)
    j = int(np.argmax(rel))
    i_star = start + j + 1
    k = n - (i_star + 1)
    k = int(max(2, min(12, k)))
    return k, i_star, rel, start

k_opt, i_star, rel_jumps, rel_start = choose_k_via_relative_jump(Z_full, tail=120)
labels_full = fcluster(Z_full, t=k_opt, criterion="maxclust")

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels_full, s=8)
ax.set_title(f"Q7) Swiss roll — Ward linkage — clusters (k={k_opt})")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x3")
plt.show()

unique, counts = np.unique(labels_full, return_counts=True)
print("Q7) Tamanhos dos clusters (k =", k_opt, "):")
for u, c in zip(unique, counts):
    print(f"  Cluster {u}: {c} pontos")


# -----------------------------
# 8) Dendrogram
# -----------------------------
subset_size = 200
subset_idx = np.random.choice(n_samples_full, size=subset_size, replace=False)
X_sub = X[subset_idx]

Z_sub = linkage(X_sub, method="ward", metric="euclidean")

plt.figure(figsize=(8, 5))
dendro = dendrogram(Z_sub, no_labels=True)
plt.title("Q8) Dendrograma (Ward) — subconjunto de 200 pontos")
plt.xlabel("Amostras (reordenadas)")
plt.ylabel("Distância de ligação")
plt.show()


# -----------------------------
# 9) Heatmap associado ao dendrograma
# -----------------------------
dendro_silent = dendrogram(Z_sub, no_labels=True, no_plot=True)
leaves_order = dendro_silent["leaves"]

D = squareform(pdist(X_sub, metric="euclidean"))
D_ordered = D[np.ix_(leaves_order, leaves_order)]

plt.figure(figsize=(6, 6))
plt.imshow(D_ordered, aspect="auto")
plt.title("Q9) Heatmap de distâncias — ordenado pelas folhas do dendrograma")
plt.xlabel("Amostras (ordem do dendrograma)")
plt.ylabel("Amostras (ordem do dendrograma)")
plt.colorbar(label="Distância Euclidiana")
plt.show()


# -----------------------------
# 11) Demonstração: determinar k pela “salta” relativa de distâncias
# -----------------------------
d_full = Z_full[:, 2]
m = len(d_full)
tail = min(120, m - 1)
start = max(0, m - tail - 1)
d_tail = d_full[start:]
rel = np.diff(d_tail) / np.maximum(d_tail[:-1], 1e-12)
j = int(np.argmax(rel))
i_star_plot = start + j + 1
k_demo = int(max(2, min(12, X.shape[0] - (i_star_plot + 1))))

plt.figure(figsize=(8, 4))
plt.plot(np.arange(start, m), d_full[start:], marker="o", linewidth=1)
plt.axvline(i_star_plot, linestyle="--")
plt.title(f"Q11) Distâncias de ligação (últimos {tail} passos) — salto indica k≈{k_demo}")
plt.xlabel("Índice do merge")
plt.ylabel("Distância")
plt.show()

print(f"Q11) Heurística do salto relativo sugeriu k ≈ {k_demo} (linha tracejada na figura).")

# -----------------------------
# 12) Avaliação do k sugerido vs. resultado da Q7
# -----------------------------
print(f"Q12) Comparação: k sugerido (≈{k_demo}) vs. k usado na Q7 ({k_opt}).")

