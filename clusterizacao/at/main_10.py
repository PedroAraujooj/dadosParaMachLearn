from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

# ---------------------------
# Utilidades de IO e helpers
# ---------------------------

def find_file(candidates):
    """Retorna o primeiro caminho existente na lista de candidates (Path)."""
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Arquivo não encontrado em: {candidates}")

def load_olivetti():
    """
    Tenta carregar os .npy de:
      ./olivetti/
      ./
      /mnt/data/
    Retorna:
      X_imgs: (n_samples, 64, 64) float32/float64
      y: (n_samples,) int
    """
    base_candidates = [
        Path(__file__).parent / "olivetti",
        Path.cwd() / "olivetti",
        Path.cwd(),
        Path("/mnt/data"),
        ]
    faces_name = "olivetti_faces.npy"
    target_name = "olivetti_faces_target.npy"

    faces_path = find_file([b / faces_name for b in base_candidates])
    target_path = find_file([b / target_name for b in base_candidates])

    X_imgs = np.load(faces_path)  # (n, 64, 64)
    y = np.load(target_path)      # (n,)
    return X_imgs, y

def to_vectors(X_imgs):
    """Achata imagens 64x64 -> vetor 4096."""
    n, h, w = X_imgs.shape
    return X_imgs.reshape(n, h * w)

def safe_silhouette(X, labels):
    """
    Silhouette é definida apenas quando há >= 2 rótulos diferentes
    e sem o caso degenerado (todo mundo no mesmo cluster ou -1).
    """
    unique = np.unique(labels)
    # Se só 1 cluster (ou apenas ruído) não dá para calcular
    if len(unique) < 2:
        return np.nan
    # Se for DBSCAN com tudo -1 (ruído), também não dá
    if len(unique) == 1 and unique[0] == -1:
        return np.nan
    # Para distância, usamos euclidiana no espaço usado no ajuste
    try:
        return float(silhouette_score(X, labels, metric="euclidean"))
    except Exception:
        return np.nan

def print_metrics(name, y_true, y_pred, X_for_sil):
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    sil = safe_silhouette(X_for_sil, y_pred)
    print(f"\n=== {name} ===")
    print(f"Clusters únicos (previstos): {len(np.unique(y_pred))}")
    print(f"ARI: {ari:.4f} | NMI: {nmi:.4f} | Silhouette: {sil if np.isnan(sil) else round(sil, 4)}")

# ---------------------------
# Visualizações
# ---------------------------

def scatter_2d(X_2d, labels, title):
    """Dispersão 2D (PCA) colorida pelo rótulo previsto."""
    plt.figure(figsize=(7, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, s=20)
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.show()

def montage_per_cluster(X_imgs, labels, per_cluster=6, title="Montagem por cluster"):
    """
    Monta uma grade de rostos (amostra) por cluster.
    - Mostra até 'per_cluster' imagens de cada cluster em sequência.
    - Para muitos clusters (ex: 40), a grade pode ficar grande; então
      mostramos no máximo 8 clusters por figura para manter legível.
    """
    unique = [c for c in np.unique(labels) if c != -1]  # ignora ruído
    if len(unique) == 0:
        print("Sem clusters (ou apenas ruído) para montar imagens.")
        return

    h, w = X_imgs.shape[1], X_imgs.shape[2]
    clusters = sorted(unique)

    # Limitar clusters por página para não estourar
    clusters_per_page = 8
    for page_start in range(0, len(clusters), clusters_per_page):
        page_clusters = clusters[page_start:page_start + clusters_per_page]
        n_rows = len(page_clusters)
        n_cols = per_cluster

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6*n_cols, 1.6*n_rows))
        if n_rows == 1:
            axes = np.expand_dims(axes, 0)  # garante 2D

        for i, c in enumerate(page_clusters):
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                # cluster vazio (improvável), pula
                for j in range(n_cols):
                    axes[i, j].axis('off')
                continue
            # amostra até per_cluster
            if len(idx) > per_cluster:
                np.random.seed(42)
                idx = np.random.choice(idx, per_cluster, replace=False)

            # preenche a linha
            for j in range(n_cols):
                ax = axes[i, j]
                if j < len(idx):
                    ax.imshow(X_imgs[idx[j]], cmap="gray")
                    ax.axis('off')
                else:
                    ax.axis('off')
            axes[i, 0].set_ylabel(f"Cluster {c}", rotation=0, labelpad=40, va="center")

        fig.suptitle(f"{title} (clusters {page_start+1}–{page_start+len(page_clusters)})")
        plt.tight_layout()
        plt.show()

# ---------------------------
# Pipeline principal
# ---------------------------

def main():
    print("Carregando Olivetti Faces...")
    X_imgs, y = load_olivetti()
    n_samples, H, W = X_imgs.shape
    n_classes = len(np.unique(y))
    print(f"Amostras: {n_samples} | Tamanho: {H}x{W} | Classes (sujeitos): {n_classes}")

    # Vetorização e pré-processamento
    X = to_vectors(X_imgs)                     # (n, 4096)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_std = scaler.fit_transform(X)            # padroniza

    # PCA para reduzir dimensionalidade (ajuda KMeans, Agglo, DBSCAN)
    pca_dim = 50
    pca = PCA(n_components=pca_dim, random_state=42)
    X_pca = pca.fit_transform(X_std)

    # 2D para visualização
    pca2 = PCA(n_components=2, random_state=42)
    X_2d = pca2.fit_transform(X_std)

    # ---------------------------
    # K-Means
    # ---------------------------
    k = n_classes  # 40
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    y_km = km.fit_predict(X_pca)
    print_metrics("K-Means (k=40)", y, y_km, X_pca)
    scatter_2d(X_2d, y_km, "Dispersão 2D (PCA) - K-Means (k=40)")
    montage_per_cluster(X_imgs, y_km, per_cluster=6, title="K-Means: amostra por cluster")

    # ---------------------------
    # Agglomerative (Ward)
    # ---------------------------
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    y_agg = agg.fit_predict(X_pca)  # Ward requer dist euclidiana; ok no espaço PCA
    print_metrics("Agglomerative (Ward, k=40)", y, y_agg, X_pca)
    scatter_2d(X_2d, y_agg, "Dispersão 2D (PCA) - Agglomerative (Ward, k=40)")
    montage_per_cluster(X_imgs, y_agg, per_cluster=6, title="Agglomerative (Ward): amostra por cluster")

    # ---------------------------
    # DBSCAN (busca simples de hiperparâmetros)
    # ---------------------------
    eps_grid = [2.0, 3.0, 4.0, 5.0, 6.0]        # em espaço PCA padronizado → magnitudes razoáveis
    min_samples_grid = [3, 5, 8, 10]
    results = np.zeros((len(eps_grid), len(min_samples_grid)), dtype=float)

    best_score = -1.0
    best_params = None
    best_pred = None

    print("\nBuscando hiperparâmetros para DBSCAN (ARI)...")
    for i, eps in enumerate(eps_grid):
        for j, ms in enumerate(min_samples_grid):
            db = DBSCAN(eps=eps, min_samples=ms, metric="euclidean")
            y_db = db.fit_predict(X_pca)
            # Se todo mundo for ruído (-1), ARI ~ 0; ainda assim avaliamos
            ari = adjusted_rand_score(y, y_db)
            results[i, j] = ari
            if ari > best_score and len(np.unique(y_db)) > 1:
                best_score = ari
                best_params = (eps, ms)
                best_pred = y_db

    print("Matriz ARI (linhas=eps, colunas=min_samples):")
    print("      ", "  ".join([f"ms={ms:>2d}" for ms in min_samples_grid]))
    for i, eps in enumerate(eps_grid):
        row = "  ".join([f"{results[i, j]:.3f}" for j in range(len(min_samples_grid))])
        print(f"eps={eps:<3} {row}")

    # Heatmap simples
    try:
        plt.figure(figsize=(6, 4))
        plt.imshow(results, aspect='auto')
        plt.xticks(range(len(min_samples_grid)), min_samples_grid)
        plt.yticks(range(len(eps_grid)), eps_grid)
        plt.xlabel("min_samples")
        plt.ylabel("eps")
        plt.title("DBSCAN - ARI por (eps, min_samples)")
        plt.colorbar(label="ARI")
        plt.tight_layout()
        plt.show()
    except Exception:
        pass

    if best_pred is None:
        # Como fallback, roda um DBSCAN "padrão" razoável
        print("\nNão foi encontrado conjunto de hiperparâmetros com mais de 1 cluster.")
        print("Executando DBSCAN com eps=5.0, min_samples=5 como referência.")
        db = DBSCAN(eps=5.0, min_samples=5, metric="euclidean")
        best_pred = db.fit_predict(X_pca)
        best_params = (5.0, 5)

    eps_best, ms_best = best_params
    print(f"\nMelhores parâmetros DBSCAN (p/ ARI): eps={eps_best}, min_samples={ms_best}")
    print_metrics(f"DBSCAN (eps={eps_best}, min_samples={ms_best})", y, best_pred, X_pca)
    scatter_2d(X_2d, best_pred, f"Dispersão 2D (PCA) - DBSCAN (eps={eps_best}, ms={ms_best})")
    montage_per_cluster(X_imgs, best_pred, per_cluster=6, title=f"DBSCAN: amostra por cluster (eps={eps_best}, ms={ms_best})")

    # ---------------------------
    # Comparação rápida
    # ---------------------------
    print("\n=== Comparação rápida (ARI / NMI / Silhouette) ===")
    names = ["K-Means (k=40)", "Agglomerative (Ward, k=40)", f"DBSCAN (eps={eps_best}, ms={ms_best})"]
    preds = [y_km, y_agg, best_pred]
    for nm, pr in zip(names, preds):
        ari = adjusted_rand_score(y, pr)
        nmi = normalized_mutual_info_score(y, pr)
        sil = safe_silhouette(X_pca, pr)
        print(f"{nm:30s} | ARI={ari:.4f} | NMI={nmi:.4f} | Sil={sil if np.isnan(sil) else round(sil, 4)}")

if __name__ == "__main__":
    main()
