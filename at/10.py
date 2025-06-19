import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

DATA_PATH = Path(__file__).with_name("cancerdataset.csv")
df = pd.read_csv(DATA_PATH)

df["PULMONARY_DISEASE"] = df["PULMONARY_DISEASE"].map({"YES": 1, "NO": 0})

X = df.drop(columns="PULMONARY_DISEASE")
y = df["PULMONARY_DISEASE"]

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# PCA ============================================
pca = PCA().fit(X_std)
explained = pca.explained_variance_ratio_
cum_explained = np.cumsum(explained)

k = np.argmax(cum_explained >= 0.95) + 1
print(f"Componentes necessárias para 95 % da variância: {k}")

pca_k = PCA(n_components=k)
X_pca = pca_k.fit_transform(X_std)

plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
components = np.arange(1, len(explained) + 1)
plt.bar(components, explained, alpha=0.7, label="Variância por componente")
plt.step(components, cum_explained, where="mid", label="Variância acumulada")
plt.xlabel("Componente principal")
plt.ylabel("Proporção da variância")
plt.title("Scree plot")
plt.legend()

if k >= 2:
    plt.subplot(1, 2, 2)
    colors = np.array(["tab:blue", "tab:red"])[y.values]  # 0 = azul, 1 = vermelho
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, edgecolor="k")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("PC1 × PC2 por classe\n(azul = não, vermelho = sim)")

plt.tight_layout()
plt.show()
