import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from time import time
from collections import Counter


# ----------------------------
# Helpers
# ----------------------------
def print_top_terms(model, feature_names, n_top=15, header="Tópicos"):
    print(f"\n=== {header} (top {n_top} termos) ===")
    if hasattr(model, "components_"):
        for i, comp in enumerate(model.components_):
            top_idx = comp.argsort()[::-1][:n_top]
            terms = [feature_names[j] for j in top_idx]
            print(f"#{i:02d}: " + ", ".join(terms))
    else:
        raise ValueError("Modelo sem atributo components_ para imprimir termos.")

def print_top_terms_kmeans(km, feature_names, n_top=15, header="Clusters K-Means"):
    print(f"\n=== {header} (top {n_top} termos) ===")
    centers = km.cluster_centers_
    for i in range(centers.shape[0]):
        top_idx = centers[i].argsort()[::-1][:n_top]
        terms = [feature_names[j] for j in top_idx]
        print(f"Cluster {i:02d}: " + ", ".join(terms))

def evaluate_kmeans(X, labels_true, labels_pred, metric_sample=10000):
    print("\n=== Avaliação K-Means ===")
    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    homo = homogeneity_score(labels_true, labels_pred)
    comp = completeness_score(labels_true, labels_pred)
    vmea = v_measure_score(labels_true, labels_pred)
    print(f"ARI: {ari:.4f}")
    print(f"NMI: {nmi:.4f}")
    print(f"Homogeneidade: {homo:.4f}")
    print(f"Completude: {comp:.4f}")
    print(f"V-Measure: {vmea:.4f}")

    # Silhouette pode ser caro; usar amostra e métrica cosine (bom p/ TF-IDF)
    n = X.shape[0]
    if n > metric_sample:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, size=metric_sample, replace=False)
        X_s = X[idx]
        y_s = labels_pred[idx]
    else:
        X_s = X
        y_s = labels_pred
    try:
        sil = silhouette_score(X_s, y_s, metric="cosine")
        print(f"Silhouette (cosine, amostra={X_s.shape[0]}): {sil:.4f}")
    except Exception as e:
        print(f"Silhouette não calculado: {e}")

def plot_tsne(X_tfidf, labels_pred, labels_true, title_left="K-Means (clusters)", title_right="Rótulos verdadeiros"):
    print("\nGerando visualização 2D (pode demorar um pouco)...")
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_50 = svd.fit_transform(X_tfidf)
    X_50 = normalize(X_50)
    tsne = TSNE(n_components=2, init="pca", random_state=42, perplexity=40, learning_rate="auto")
    X_2d = tsne.fit_transform(X_50)

    fig = plt.figure(figsize=(12, 5))

    # Plot clusters
    ax1 = fig.add_subplot(1, 2, 1)
    sc1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_pred, s=6, alpha=0.7)
    ax1.set_title(title_left)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Plot verdade
    ax2 = fig.add_subplot(1, 2, 2)
    sc2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_true, s=6, alpha=0.7)
    ax2.set_title(title_right)
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.tight_layout()
    plt.show()

def contingency_heatmap(labels_true, labels_pred, title="Matriz de Contingência (verdade x cluster)"):
    # constrói matriz (#classes x #clusters)
    n_true = int(labels_true.max()) + 1
    n_pred = int(labels_pred.max()) + 1
    mat = np.zeros((n_true, n_pred), dtype=int)
    for t, p in zip(labels_true, labels_pred):
        mat[t, p] += 1

    plt.figure(figsize=(8, 6))
    plt.imshow(mat, aspect='auto')
    plt.title(title)
    plt.xlabel("Cluster")
    plt.ylabel("Classe verdadeira")
    plt.colorbar(label="Contagem")
    plt.tight_layout()
    plt.show()

# ----------------------------
# 1) Carregar dataset
# ----------------------------
print("Carregando dataset 20 Newsgroups...")
t0 = time()
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
data = newsgroups.data
target = np.array(newsgroups.target)
target_names = newsgroups.target_names
print(f"Tempo: {time()-t0:.2f}s | Amostras: {len(data)} | Classes: {len(target_names)}")

# ----------------------------
# 2) Vetorização (TF-IDF)
# ----------------------------
print("\nVetorizando com TF-IDF...")
tfidf = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
X_tfidf = tfidf.fit_transform(data)
tfidf_terms = np.array(tfidf.get_feature_names_out())
print(f"Shape TF-IDF: {X_tfidf.shape}")

# ----------------------------
# 3) K-Means
# ----------------------------
k = len(target_names)  # 20 clusters
print(f"\nTreinando K-Means (k={k})...")
t0 = time()
kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init=20, max_iter=200)
labels_km = kmeans.fit_predict(X_tfidf)
print(f"K-Means treinado em {time()-t0:.2f}s")

# Resultados K-Means
evaluate_kmeans(X_tfidf, target, labels_km, metric_sample=10000)
print_top_terms_kmeans(kmeans, tfidf_terms, n_top=15, header="Clusters K-Means (termos mais representativos)")

# Visualizações K-Means
contingency_heatmap(target, labels_km, title="Contingência: Classe verdadeira x Cluster (K-Means)")
plot_tsne(X_tfidf, labels_km, target, title_left="K-Means (clusters)", title_right="Classes verdadeiras")

# ----------------------------
# 4) NMF (tópicos)
# ----------------------------
n_topics = 20
print(f"\nTreinando NMF para {n_topics} tópicos (sobre TF-IDF)...")
t0 = time()
nmf = NMF(n_components=n_topics, init='nndsvda', random_state=42, max_iter=400, alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0)
W = nmf.fit_transform(X_tfidf)
H = nmf.components_
print(f"NMF treinado em {time()-t0:.2f}s")
print_top_terms(nmf, tfidf_terms, n_top=15, header="NMF Tópicos")

# opcional: distribuição de documentos por tópico dominante
top_topic_per_doc = W.argmax(axis=1)
counts = Counter(top_topic_per_doc)
print("\nDistribuição de documentos por tópico (NMF):")
for t_id, cnt in sorted(counts.items()):
    print(f"Tópico {t_id:02d}: {cnt}")

# ----------------------------
# 5) LDA (tópicos)
# ----------------------------
print("\nVetorizando com contagem (para LDA)...")
count_vect = CountVectorizer(max_df=0.5, min_df=2, stop_words='english')
X_counts = count_vect.fit_transform(data)
count_terms = np.array(count_vect.get_feature_names_out())
print(f"Shape Counts: {X_counts.shape}")

print(f"\nTreinando LDA para {n_topics} tópicos (sobre contagens)...")
t0 = time()
lda = LatentDirichletAllocation(
    n_components=n_topics,
    learning_method='batch',
    max_iter=20,
    random_state=42,
    evaluate_every=0,
    doc_topic_prior=None,  # usa padrão 1/n_components
    topic_word_prior=None  # usa padrão 1/n_components
)
lda.fit(X_counts)
print(f"LDA treinado em {time()-t0:.2f}s")
print_top_terms(lda, count_terms, n_top=15, header="LDA Tópicos")

doc_topic = lda.transform(X_counts)  # matriz (n_docs x n_topics)
top_topic_per_doc_lda = doc_topic.argmax(axis=1)
counts_lda = Counter(top_topic_per_doc_lda)
print("\nDistribuição de documentos por tópico (LDA):")
for t_id, cnt in sorted(counts_lda.items()):
    print(f"Tópico {t_id:02d}: {cnt}")

print("\n=== FIM ===")
