import numpy as np
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

# -------------------------------------------------------------------
# 1. Carregar o dataset e identificar features contínuas
# -------------------------------------------------------------------
dados = load_breast_cancer()
X = pd.DataFrame(dados.data, columns=dados.feature_names)

print("=== 1. Features contínuas ===")
print(X.columns.tolist())

# -------------------------------------------------------------------
# 2. Discretização com bins fixos (largura uniforme)
# -------------------------------------------------------------------
feat1, feat2 = "mean radius", "mean texture"
selecionadas = X[[feat1, feat2]]

disc_fixos = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
bins_fixos = disc_fixos.fit_transform(selecionadas)
df_fixos = pd.DataFrame(bins_fixos, columns=[f"{feat1}_fixo", f"{feat2}_fixo"])

print("\n=== 2. Contagem por bin (fixos) ===")
print(df_fixos.apply(pd.Series.value_counts).fillna(0).astype(int))

# -------------------------------------------------------------------
# 3. Discretização com bins variáveis (quantis)
# -------------------------------------------------------------------
disc_var = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
bins_var = disc_var.fit_transform(selecionadas)
df_var = pd.DataFrame(bins_var, columns=[f"{feat1}_quantil", f"{feat2}_quantil"])

print("\n=== 3. Contagem por bin (quantis) ===")
print(df_var.apply(pd.Series.value_counts).fillna(0).astype(int))

# -------------------------------------------------------------------
# 4. Normalização personalizada com FunctionTransformer
# -------------------------------------------------------------------
def normaliza(arr):
    return (arr - arr.mean(axis=0)) / (arr.max(axis=0) - arr.min(axis=0))

transf_custom = FunctionTransformer(normaliza)
X_custom = pd.DataFrame(transf_custom.fit_transform(X), columns=X.columns)

# -------------------------------------------------------------------
# 5. PowerTransform
# -------------------------------------------------------------------
pt = PowerTransformer(method="yeo-johnson")
X_pt = pd.DataFrame(pt.fit_transform(X), columns=X.columns)

# -------------------------------------------------------------------
# 6. MinMaxScaler
# -------------------------------------------------------------------
mm = MinMaxScaler()
X_mm = pd.DataFrame(mm.fit_transform(X), columns=X.columns)

# -------------------------------------------------------------------
# 7. StandardScaler
# -------------------------------------------------------------------
ss = StandardScaler()
X_ss = pd.DataFrame(ss.fit_transform(X), columns=X.columns)

# -------------------------------------------------------------------
# 8. Normalização L2
# -------------------------------------------------------------------
l2 = Normalizer(norm="l2")
X_l2 = pd.DataFrame(l2.fit_transform(X), columns=X.columns)

# -------------------------------------------------------------------
# 9. Estatísticas de comparação (média e desvio das 5 primeiras colunas)
# -------------------------------------------------------------------
def resumo(df, nome):
    stats = df.iloc[:, :5].agg(["mean", "std"]).round(3)
    print(f"\n=== Estatísticas (primeiras 5 colunas) — {nome} ===")
    print(stats)

resumo(X, "Original")
resumo(X_custom, "Normalização personalizada")
resumo(X_pt, "PowerTransform")
resumo(X_mm, "MinMaxScaler")
resumo(X_ss, "StandardScaler")
resumo(X_l2, "L2 por amostra")
