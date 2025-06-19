import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import (
    KBinsDiscretizer,
    FunctionTransformer,
    PowerTransformer,
    MinMaxScaler,
    StandardScaler,
    Normalizer,
)
import matplotlib.pyplot as plt

dados = load_breast_cancer()
X = pd.DataFrame(dados.data, columns=dados.feature_names)

print("=== 2A. Features contínuas ===")
print(X.columns.tolist())


def plot_bins(df, column, titulo):
    counts = df[column].value_counts().sort_index()
    plt.figure()
    plt.bar(counts.index, counts.values)
    plt.xlabel("Bin")
    plt.ylabel("Número de amostras")
    plt.title(titulo)
    plt.xticks(range(len(counts)))
    plt.tight_layout()
    plt.show()


feat1, feat2 = "mean radius", "mean texture"
selecionadas = X[[feat1, feat2]]

print("\n=== 2B. Contagem por bin (fixos) ===")
disc_fixos = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
bins_fixos = disc_fixos.fit_transform(selecionadas)
df_fixos = pd.DataFrame(bins_fixos, columns=[feat1, feat2])
print(df_fixos.apply(pd.Series.value_counts).fillna(0).astype(int))
plot_bins(df_fixos, feat1, f"{feat1} — Bins fixos (uniforme)")
plot_bins(df_fixos, feat2, f"{feat2} — Bins fixos (uniforme)")

print("\n=== 2C. Contagem por bin (quantis) ===")
disc_var = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
bins_var = disc_var.fit_transform(selecionadas)
df_var = pd.DataFrame(bins_var, columns=[feat1, feat2])
plot_bins(df_var, feat1, f"{feat1} — Bins por quantil")
plot_bins(df_var, feat2, f"{feat2} — Bins por quantil")
print(df_var.apply(pd.Series.value_counts).fillna(0).astype(int))


def imprimir(df, nome):
    stats = df.iloc[:, :5].agg(["mean", "std"]).round(3)
    print(f"\n=== Estatísticas (primeiras 5 colunas) — {nome} ===")
    print(stats)


#============================================== original
imprimir(X, "Original")


def normaliza(arr):
    return (arr - arr.min(axis=0)) / (arr.max(axis=0) - arr.min(axis=0))


#=============== MinMax personalizado
transf_custom = FunctionTransformer(normaliza)
X_custom = pd.DataFrame(transf_custom.fit_transform(X), columns=X.columns)
imprimir(X_custom, "Normalização personalizada(MinMaxScaler)")

#=================================== PowerTransform
pt = PowerTransformer(method="yeo-johnson")
X_pt = pd.DataFrame(pt.fit_transform(X), columns=X.columns)
imprimir(X_pt, "PowerTransform")

#========================================= MinMaxScaler
mm = MinMaxScaler()
X_mm = pd.DataFrame(mm.fit_transform(X), columns=X.columns)
imprimir(X_mm, "MinMaxScaler")

#====================== StandardScaler
ss = StandardScaler()
X_ss = pd.DataFrame(ss.fit_transform(X), columns=X.columns)
imprimir(X_ss, "StandardScaler")

#=================================== l2
l2 = Normalizer(norm="l2")
X_l2 = pd.DataFrame(l2.fit_transform(X), columns=X.columns)
imprimir(X_l2, "L2 por amostra")
