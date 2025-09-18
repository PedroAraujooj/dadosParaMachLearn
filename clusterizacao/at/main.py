from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    pairwise_distances_argmin,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.datasets import make_swiss_roll
from scipy.cluster.hierarchy import linkage, dendrogram


# -------------------------------
# Configurações gerais
# -------------------------------
RANDOM_STATE = 42
CSV_NAME = "Mall_Customers.csv"  # Deve estar na mesma pasta deste .py


# -------------------------------
# Utilitários
# -------------------------------
def read_data() -> pd.DataFrame:
    """Lê o CSV Mall_Customers da mesma pasta do script (ou do cwd)."""
    here = Path(__file__).parent
    p = here / CSV_NAME
    if not p.exists():
        p = Path.cwd() / CSV_NAME
    if not p.exists():
        raise FileNotFoundError(f"Arquivo {CSV_NAME} não encontrado em {here} nem em {Path.cwd()}")
    df = pd.read_csv(p)
    return df


def initial_analysis(df: pd.DataFrame) -> str:
    """Texto com objetivos e análise inicial do dataset considerando K-Means."""
    linhas, colunas = df.shape
    cols = list(df.columns)
    texto = []
    texto.append("=== 1) Análise Inicial — Mall Customer Segmentation Data ===")
    texto.append(f"Dimensões: {linhas} linhas x {colunas} colunas")
    texto.append(f"Colunas: {cols}")
    texto.append("Variáveis úteis para clusterização:")
    texto.append(" - 'Age' (idade) — comportamento por faixa etária;")
    texto.append(" - 'Annual Income (k$)' (renda anual) — poder aquisitivo;")
    texto.append(" - 'Spending Score (1-100)' — perfil de gasto;")
    texto.append(" - 'Gender' pode ser codificado e incluído (impacto geralmente menor).")
    texto.append("Objetivos típicos para K-Means:")
    texto.append(" - Descobrir segmentos de clientes com perfis de consumo distintos;")
    texto.append(" - Apoiar marketing segmentado e ações de fidelização;")
    texto.append(" - Entender padrões de consumo para alocação de recursos.")
    texto.append("Observação: K-Means assume clusters mais ou menos esféricos e com variâncias semelhantes.")
    return "\n".join(texto)


def build_preprocess(df: pd.DataFrame) -> Tuple[Pipeline, List[str]]:
    """Pipeline de pré-processamento: One-Hot (Gender) + StandardScaler (numéricas)."""
    num_cols = [c for c in df.columns if c in ["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    cat_cols = [c for c in df.columns if c in ["Gender"]]

    transformers = []
    if len(cat_cols) > 0:
        transformers.append(("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), cat_cols))
    if len(num_cols) > 0:
        transformers.append(("num", StandardScaler(), num_cols))

    pre = ColumnTransformer(transformers, remainder="drop")

    # Nomes de features (opcional, informativo)
    feature_names = []
    if len(cat_cols) > 0:
        temp = ColumnTransformer(transformers, remainder="drop")
        _ = temp.fit(df)
        ohe = temp.named_transformers_.get("cat", None)
        if ohe is not None:
            feature_names += list(ohe.get_feature_names_out(cat_cols))
    feature_names += num_cols

    pipe = Pipeline([("pre", pre)])
    return pipe, feature_names


def auto_k_by_silhouette(X: np.ndarray, k_min: int = 2, k_max: int = 10) -> Tuple[int, Dict[int, float]]:
    """Seleciona k pela maior silhueta média no intervalo [k_min, k_max]."""
    scores = {}
    best_k, best_score = None, -1.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X, labels)
        else:
            score = -1.0
        scores[k] = score
        if score > best_score:
            best_k, best_score = k, score
    return best_k, scores


def plot_silhouette_scores(scores: Dict[int, float], title: str):
    ks = sorted(scores.keys())
    vals = [scores[k] for k in ks]
    plt.figure(figsize=(6, 4), dpi=140)
    plt.plot(ks, vals, marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette médio")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.close()


def run_kmeans(X: np.ndarray, k: int) -> Tuple[KMeans, np.ndarray, float]:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan
    return km, labels, sil


def quantizacao_vetorial_kmeans(X: np.ndarray, km: KMeans) -> Tuple[np.ndarray, float]:
    """Quantização vetorial: substitui cada vetor pelo centróide do seu cluster."""
    centers = km.cluster_centers_
    idx = pairwise_distances_argmin(X, centers)
    Xq = centers[idx]
    mse = float(np.mean(np.sum((X - Xq) ** 2, axis=1)))
    return Xq, mse


def run_pca(X: np.ndarray, n_components: int = 2) -> Tuple[PCA, np.ndarray, np.ndarray]:
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X2 = pca.fit_transform(X)
    return pca, X2, pca.explained_variance_ratio_


def scatter_2d(X2: np.ndarray, labels: np.ndarray | None, title: str):
    plt.figure(figsize=(6, 5), dpi=140)
    if labels is None:
        plt.scatter(X2[:, 0], X2[:, 1], s=18)
    else:
        plt.scatter(X2[:, 0], X2[:, 1], s=18, c=labels)
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.close()


def run_agglomerative(X: np.ndarray, n_clusters: int, linkage_name: str) -> Tuple[AgglomerativeClustering, np.ndarray, float]:
    ag = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_name)
    labels = ag.fit_predict(X)
    sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan
    return ag, labels, sil


def plot_dendrogram(X: np.ndarray, method: str, max_d: float | None = None):
    """Mostra dendrograma usando scipy.linkage + scipy.dendrogram."""
    Z = linkage(X, method=method)
    plt.figure(figsize=(8, 5), dpi=140)
    dendrogram(Z, leaf_rotation=90.0, leaf_font_size=8, color_threshold=max_d)
    plt.title(f"Dendrograma — linkage = {method}")
    plt.xlabel("Amostras")
    plt.ylabel("Distância")
    if max_d is not None:
        plt.axhline(y=max_d, linestyle="--")
    plt.tight_layout()
    plt.show()
    plt.close()
    return Z


def heatmap_with_row_dendrogram(X: np.ndarray, method: str, metric_label: str = "z-score"):
    """Associa dendrograma (linhas) a um mapa de calor simples (sem seaborn)."""
    Z = linkage(X, method=method)
    leaves = dendrogram(Z, no_plot=True)["leaves"]
    X_ord = X[leaves, :]

    # Normaliza colunas para z-score (melhor contraste)
    X_norm = (X_ord - X_ord.mean(axis=0, keepdims=True)) / (X_ord.std(axis=0, keepdims=True) + 1e-9)

    fig = plt.figure(figsize=(8, 6), dpi=140)
    # Dendrograma à esquerda
    ax_dend = fig.add_axes([0.05, 0.1, 0.25, 0.8])
    dendrogram(Z, orientation="left", ax=ax_dend, no_labels=True)
    ax_dend.set_xticks([])
    ax_dend.set_yticks([])

    # Heatmap à direita
    ax_heat = fig.add_axes([0.33, 0.1, 0.6, 0.8])
    im = ax_heat.imshow(X_norm, aspect="auto", interpolation="nearest")
    ax_heat.set_title(f"Mapa de calor (ordenado por linkage='{method}')")
    ax_heat.set_xlabel("Features")
    ax_heat.set_ylabel("Amostras")
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label(metric_label)

    plt.show()
    plt.close()


def explain_linkages() -> str:
    txt = []
    txt.append("=== 4) Comportamento por método de linkage ===")
    txt.append("- ward: minimiza o aumento da variância intra-cluster a cada fusão;")
    txt.append("        tende a formar grupos compactos e de tamanho semelhante (requer distância euclidiana).")
    txt.append("- average: usa a distância média entre todos os pares de pontos de dois clusters;")
    txt.append("           mais equilibrado que 'single', menos compacto que 'complete'.")
    txt.append("- single: usa a menor distância entre quaisquer pontos de dois clusters;")
    txt.append("          propenso a 'chaining' (encadeamento), clusters alongados e sensível a ruído/outliers.")
    txt.append("- complete: usa a maior distância entre quaisquer pontos de dois clusters;")
    txt.append("            favorece grupos compactos, mas pode quebrar clusters maiores na presença de outliers.")
    return "\n".join(txt)


def kmeans_failure_notes() -> str:
    txt = []
    txt.append("=== 2) Quando o K-Means pode falhar? ===")
    txt.append("- Clusters não esféricos (anel, lua crescente);")
    txt.append("- Tamanhos/densidades muito diferentes;")
    txt.append("- Outliers (centroides são sensíveis);")
    txt.append("- Escalas muito diferentes entre features (padronize!);")
    txt.append("- k inadequado.")
    return "\n".join(txt)


# -------------------------------
# DBSCAN helpers
# -------------------------------
def dbscan_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette ignorando ruído (-1). Retorna NaN se <2 clusters válidos."""
    mask = labels != -1
    if mask.sum() < 2:
        return np.nan
    lbl = labels[mask]
    if len(np.unique(lbl)) < 2:
        return np.nan
    return silhouette_score(X[mask], lbl)


def grid_dbscan(
        X: np.ndarray,
        eps_values: List[float],
        min_samples_values: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Explora uma grade de (eps, min_samples).
    Retorna matrices:
      - sil_mat: silhouette (ignorando ruído)
      - k_mat: número de clusters (sem contar -1)
      - noise_ratio_mat: proporção de ruído
    """
    sil_mat = np.full((len(min_samples_values), len(eps_values)), np.nan, dtype=float)
    k_mat = np.zeros_like(sil_mat)
    noise_ratio_mat = np.zeros_like(sil_mat)
    for i, ms in enumerate(min_samples_values):
        for j, eps in enumerate(eps_values):
            labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
            sil = dbscan_silhouette(X, labels)
            k = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = float(np.mean(labels == -1))
            sil_mat[i, j] = sil
            k_mat[i, j] = k
            noise_ratio_mat[i, j] = noise_ratio
    return sil_mat, k_mat, noise_ratio_mat


def plot_dbscan_heatmap(
        eps_values: List[float],
        min_samples_values: List[int],
        Z: np.ndarray,
        title: str,
        value_label: str
):
    """
    Heatmap simples (matplotlib) para matriz Z com
    linhas = min_samples e colunas = eps.
    """
    plt.figure(figsize=(8, 5), dpi=140)
    im = plt.imshow(Z, aspect="auto", origin="upper")
    plt.title(title)
    plt.xlabel("eps")
    plt.ylabel("min_samples")
    plt.xticks(ticks=range(len(eps_values)), labels=[f"{e:.2f}" for e in eps_values], rotation=45)
    plt.yticks(ticks=range(len(min_samples_values)), labels=min_samples_values)
    cbar = plt.colorbar(im)
    cbar.set_label(value_label)
    # Anota os valores
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            val = Z[i, j]
            if np.isfinite(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.show()
    plt.close()


def best_params_from_silhouette(
        eps_values: List[float],
        min_samples_values: List[int],
        sil_mat: np.ndarray
) -> Tuple[float, int, float]:
    """Escolhe (eps, min_samples) com maior silhouette (ignora NaNs)."""
    if np.all(np.isnan(sil_mat)):
        return eps_values[0], min_samples_values[0], np.nan
    idx = np.nanargmax(sil_mat)
    i, j = np.unravel_index(idx, sil_mat.shape)
    return eps_values[j], min_samples_values[i], sil_mat[i, j]


def pairwise_agreement(labels_a: np.ndarray, labels_b: np.ndarray) -> Tuple[float, float, float, float]:
    """
    ARI e NMI entre dois rótulos, ignorando índices onde algum é -1 (ruído).
    Também retorna a cobertura (proporção de pontos usados) e #pontos usados.
    """
    mask = (labels_a != -1) & (labels_b != -1)
    if mask.sum() < 2 or len(np.unique(labels_a[mask])) < 2 or len(np.unique(labels_b[mask])) < 2:
        return np.nan, np.nan, mask.mean(), mask.sum()
    ari = adjusted_rand_score(labels_a[mask], labels_b[mask])
    nmi = normalized_mutual_info_score(labels_a[mask], labels_b[mask])
    return ari, nmi, mask.mean(), mask.sum()


# -------------------------------
# Main
# -------------------------------
def main():
    # 1) Análise inicial
    df = read_data()
    df = df.rename(columns={
        "Annual Income (k$)": "Annual Income (k$)",
        "Spending Score (1-100)": "Spending Score (1-100)",
        "CustomerID": "CustomerID",
        "Gender": "Gender",
        "Age": "Age"
    })
    analysis_text = initial_analysis(df)
    print(analysis_text)

    # 2) Pré-processamento e K-Means (seleção automática de k por silhueta)
    pre, _ = build_preprocess(df)
    X = pre.fit_transform(df)
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)

    best_k, scores = auto_k_by_silhouette(X, 2, 10)
    print("\n=== 2) K-Means — Seleção de k por Silhouette ===")
    print(f"k* (silhouette): {best_k}")
    print("Silhouette por k:", ", ".join([f"{k}:{scores[k]:.3f}" for k in sorted(scores.keys())]))
    plot_silhouette_scores(scores, "K-Means — Silhouette por k")

    km, labels_kmeans, sil_kmeans = run_kmeans(X, best_k)
    print(f"Silhouette (K-Means, k={best_k}): {sil_kmeans:.3f}")

    # PCA para visualização (2D) dos clusters do K-Means
    _, X2, _ = run_pca(X, n_components=2)
    scatter_2d(X2, labels_kmeans, f"K-Means (k={best_k}) na projeção PCA 2D")

    # 3a) Quantização vetorial com K-Means
    Xq, mse_q = quantizacao_vetorial_kmeans(X, km)
    _, X2_q, _ = run_pca(Xq, n_components=2)
    scatter_2d(X2_q, labels_kmeans, "Quantização Vetorial por K-Means (PCA 2D)")
    print("\n=== 3) Redução de Dimensionalidade e Quantização Vetorial ===")
    print(f"MSE da quantização vetorial (K-Means): {mse_q:.4f}")

    # 3b) PCA (variância explicada)
    _, _, vr_full = run_pca(X, n_components=min(3, X.shape[1]))
    plt.figure(figsize=(6, 4), dpi=140)
    plt.bar(range(1, len(vr_full) + 1), vr_full)
    plt.xlabel("Componente Principal")
    plt.ylabel("Variância explicada")
    plt.title("PCA — Variância explicada por componente")
    plt.tight_layout()
    plt.show()
    plt.close()

    # 4) Agglomerative Clustering com diferentes linkages
    linkages = ["ward", "average", "single", "complete"]
    print("\n=== 4) Agglomerative Clustering (ward, average, single, complete) ===")
    ag_labels = {}
    ag_sils = {}
    for link in linkages:
        _, labels_ag, sil_ag = run_agglomerative(X, n_clusters=best_k, linkage_name=link)
        ag_labels[link] = labels_ag
        ag_sils[link] = sil_ag
        print(f"- linkage={link:8s} | silhouette (k={best_k}): {sil_ag:.3f}")
        scatter_2d(X2, labels_ag, f"Agglomerative (linkage={link}) na PCA 2D")

        # 5) Dendrograma do linkage
        plot_dendrogram(X, method=link)

        # 6) Heatmap + dendrograma (linhas) para cada linkage
        heatmap_with_row_dendrogram(X, method=link)

    # Textos explicativos
    print("\n" + kmeans_failure_notes())
    print("\n" + explain_linkages())
    print("\n=== 5) Dendrogramas ===")
    print("Observe a altura dos cortes horizontais sugerindo um número de clusters.")
    print("\n=== 6) Mapa de Calor com Dendrograma ===")
    print("As amostras são reordenadas pelo dendrograma, revelando blocos (padrões) por feature.")

    # -------------------------------
    # 7) DBSCAN: make_swiss_roll
    # -------------------------------
    print("\n=== 7) DBSCAN no make_swiss_roll ===")
    X_sw, t = make_swiss_roll(n_samples=1200, noise=0.05, random_state=RANDOM_STATE)
    # Padroniza (DBSCAN depende de escala)
    X_sw = StandardScaler().fit_transform(X_sw)

    # Grade de parâmetros (ajuste conforme necessário)
    eps_values = list(np.linspace(0.3, 1.2, 10))
    min_samples_values = [3, 5, 8, 10, 12, 15]

    sil_mat, k_mat, noise_mat = grid_dbscan(X_sw, eps_values, min_samples_values)
    plot_dbscan_heatmap(eps_values, min_samples_values, sil_mat, "DBSCAN Swiss Roll — Silhouette (sem ruído)", "silhouette")
    plot_dbscan_heatmap(eps_values, min_samples_values, k_mat, "DBSCAN Swiss Roll — #clusters (sem ruído)", "#clusters")
    plot_dbscan_heatmap(eps_values, min_samples_values, noise_mat, "DBSCAN Swiss Roll — Proporção de ruído", "ratio ruído")

    eps_best_sw, ms_best_sw, sil_best_sw = best_params_from_silhouette(eps_values, min_samples_values, sil_mat)
    print(f"Melhor (eps, min_samples) por silhouette: ({eps_best_sw:.3f}, {ms_best_sw}) | silhouette={sil_best_sw:.3f}")

    labels_sw = DBSCAN(eps=eps_best_sw, min_samples=ms_best_sw).fit_predict(X_sw)
    # Visualização: PCA 2D para Swiss Roll (apenas para plot)
    _, Xsw2, _ = run_pca(X_sw, n_components=2)
    scatter_2d(Xsw2, labels_sw, f"Swiss Roll — DBSCAN (eps={eps_best_sw:.2f}, ms={ms_best_sw})")

    n_clusters_sw = len(set(labels_sw)) - (1 if -1 in labels_sw else 0)
    print(f"#clusters (sem ruído): {n_clusters_sw} | ruído: {(labels_sw == -1).mean():.2%}")

    # -------------------------------
    # 8) DBSCAN: Mall Customer Segmentation Data
    # -------------------------------
    print("\n=== 8) DBSCAN no Mall Customer Segmentation Data ===")
    # Tenta uma escala de eps relativa ao espaço pós-padronização:
    # Uma heurística: olhar mediana de distâncias aos 5 vizinhos mais próximos e varrer em torno.
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=6).fit(X)
    distances, _ = nbrs.kneighbors(X)
    # distância ao 5º vizinho (índice 5-1)
    kdist = np.sort(distances[:, -1])
    eps_base = np.median(kdist)
    eps_values_mall = list(np.linspace(0.5*eps_base, 1.8*eps_base, 10))
    min_samples_values_mall = [3, 4, 5, 6, 8, 10]

    sil_mat_m, k_mat_m, noise_mat_m = grid_dbscan(X, eps_values_mall, min_samples_values_mall)
    plot_dbscan_heatmap(eps_values_mall, min_samples_values_mall, sil_mat_m, "DBSCAN Mall — Silhouette (sem ruído)", "silhouette")
    plot_dbscan_heatmap(eps_values_mall, min_samples_values_mall, k_mat_m, "DBSCAN Mall — #clusters (sem ruído)", "#clusters")
    plot_dbscan_heatmap(eps_values_mall, min_samples_values_mall, noise_mat_m, "DBSCAN Mall — Proporção de ruído", "ratio ruído")

    eps_best_m, ms_best_m, sil_best_m = best_params_from_silhouette(eps_values_mall, min_samples_values_mall, sil_mat_m)
    labels_dbscan_mall = DBSCAN(eps=eps_best_m, min_samples=ms_best_m).fit_predict(X)
    _, X2_db, _ = run_pca(X, n_components=2)
    scatter_2d(X2_db, labels_dbscan_mall, f"Mall — DBSCAN (eps={eps_best_m:.3f}, ms={ms_best_m})")
    sil_dbscan_mall = dbscan_silhouette(X, labels_dbscan_mall)
    print(f"Melhor (eps, min_samples) por silhouette: ({eps_best_m:.3f}, {ms_best_m}) | silhouette={sil_best_m:.3f}")
    print(f"Silhouette (DBSCAN Mall, ignorando ruído): {sil_dbscan_mall:.3f}")
    print(f"#clusters DBSCAN (sem ruído): {len(set(labels_dbscan_mall)) - (1 if -1 in labels_dbscan_mall else 0)} | "
          f"ruído: {(labels_dbscan_mall == -1).mean():.2%}")

    # -------------------------------
    # 9) Comparação: KMeans (Q2) vs Ward (Q4) vs DBSCAN (Q8)
    # -------------------------------
    print("\n=== 9) Comparação de Algoritmos (Mall) — ARI, NMI e Silhouette ===")
    labels_ward = ag_labels["ward"]
    # Silhouette de cada (tomando cuidado no DBSCAN)
    sil_ward = silhouette_score(X, labels_ward) if len(np.unique(labels_ward)) > 1 else np.nan
    print(f"Silhouette — KMeans={sil_kmeans:.3f} | Ward={sil_ward:.3f} | DBSCAN(ign. ruído)={sil_dbscan_mall:.3f}")

    # ARI/NMI (pares), ignorando ruído no DBSCAN:
    pairs = [
        ("KMeans vs Ward", labels_kmeans, labels_ward),
        ("KMeans vs DBSCAN", labels_kmeans, labels_dbscan_mall),
        ("Ward   vs DBSCAN", labels_ward,   labels_dbscan_mall),
    ]
    for name, a, b in pairs:
        ari, nmi, cov, used = pairwise_agreement(a, b)
        print(f"{name:17s} | ARI={ari:.3f} | NMI={nmi:.3f} | cobertura={cov:.2%} | n usados={int(used)}")


if __name__ == "__main__":
    main()
