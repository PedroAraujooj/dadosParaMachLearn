from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

path = Path(__file__).parent / "shopmania.csv"


df = pd.read_csv(
    path,
    header=None,
    names=["id", "title", "category_id", "category"],
    encoding="utf-8",
    engine="python"
).dropna(subset=["title"])

X = TfidfVectorizer(stop_words="english", min_df=2).fit_transform(df["title"])

k = 3
labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
df["cluster"] = labels

amostra = df.sample(n=min(int(len(df)/10), len(df)), random_state=42)
X_amostra = X[amostra.index]
X_2d = PCA(n_components=2).fit_transform(X_amostra.toarray())

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=amostra["cluster"], cmap="tab10", s=30)
plt.title(f"Amostra visual dos clusters (k={k})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

labels_pca2d = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_2d)

plt.figure()
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_pca2d, cmap="tab10", s=30)
plt.title(f"Amostra visual (K-Means no X_2d, k={k})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

