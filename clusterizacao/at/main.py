from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances_argmin
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
        score = silhouette_score(X, labels) if k > 1 else -1.0
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
    sil = silhouette_score(X, labels) if k > 1 else np.nan
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
    sil = silhouette_score(X, labels) if n_clusters > 1 else np.nan
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

    # 2) Pré-processamento e K-Means (seleção automática de k por silhueta)
    pre, _ = build_preprocess(df)
    X = pre.fit_transform(df)
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)

    best_k, scores = auto_k_by_silhouette(X, 2, 10)
    print(analysis_text)
    print("\n=== 2) K-Means — Seleção de k por Silhouette ===")
    print(f"k* (silhouette): {best_k}")
    print("Silhouette por k:", ", ".join([f"{k}:{scores[k]:.3f}" for k in sorted(scores.keys())]))

    plot_silhouette_scores(scores, "K-Means — Silhouette por k")

    km, labels, sil = run_kmeans(X, best_k)
    print(f"Silhouette (k*): {sil:.3f}")

    # PCA para visualização (2D) dos clusters do K-Means
    _, X2, _ = run_pca(X, n_components=2)
    scatter_2d(X2, labels, f"K-Means (k={best_k}) na projeção PCA 2D")

    # 3a) Quantização vetorial com K-Means
    Xq, mse_q = quantizacao_vetorial_kmeans(X, km)
    _, X2_q, _ = run_pca(Xq, n_components=2)
    scatter_2d(X2_q, labels, "Quantização Vetorial por K-Means (PCA 2D)")
    print("\n=== 3) Redução de Dimensionalidade e Quantização Vetorial ===")
    print(f"MSE da quantização vetorial (K-Means): {mse_q:.4f}")

    # 3b) PCA (variância explicada)
    pca_full, _, vr_full = run_pca(X, n_components=min(3, X.shape[1]))
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
    for link in linkages:
        _, labels_ag, sil_ag = run_agglomerative(X, n_clusters=best_k, linkage_name=link)
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


if __name__ == "__main__":
    main()
