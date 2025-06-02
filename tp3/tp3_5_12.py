import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from category_encoders import HelmertEncoder
from sklearn.feature_extraction import FeatureHasher

data = {
    "Dia": [
        "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14"
    ],
    "Aspecto": ["Sol", "Sol", "Nuvens", "Chuva", "Chuva", "Chuva", "Nuvens", "Sol",
                "Sol", "Chuva", "Sol", "Nuvens", "Nuvens", "Chuva"],
    "Temp.": ["Quente", "Quente", "Quente", "Ameno", "Fresco", "Fresco", "Fresco",
              "Ameno", "Fresco", "Ameno", "Ameno", "Ameno", "Quente", "Ameno"],
    "Humidade": ["Elevada", "Elevada", "Elevada", "Elevada", "Normal", "Normal",
                 "Normal", "Elevada", "Normal", "Normal", "Normal", "Elevada",
                 "Normal", "Elevada"],
    "Vento": ["Fraco", "Forte", "Fraco", "Fraco", "Fraco", "Forte", "Fraco",
              "Fraco", "Fraco", "Forte", "Forte", "Forte", "Fraco", "Forte"],
    "Jogar?": ["Não", "Não", "Sim", "Sim", "Sim", "Não", "Sim", "Sim", "Sim",
               "Sim", "Sim", "Sim", "Sim", "Não"],
}
df = pd.DataFrame(data).set_index("Dia")

y = df.pop("Jogar?").map({"Não": 0, "Sim": 1})

print("=== Dataset original ===")
print(df, "\n")

# -------------------------------------------------------------------------------------------------------
# Questão 5
ohe = OneHotEncoder(sparse_output=False, dtype=int)
X_ohe = pd.DataFrame(
    ohe.fit_transform(df),
    columns=ohe.get_feature_names_out(df.columns),
    index=df.index
)

print("Q5 – One-Hot Encoding:")
print(X_ohe, "\n")

# -----------------------------------------------------------------------------
# Questão 6
X_dummy = pd.get_dummies(df,
                         drop_first=True,
                         dtype=int)

print("Q6 – Dummy Encoding:")
print(X_dummy, "\n")

# ------------------------------------------------------------------
# Questão
helmert = HelmertEncoder(cols=df.columns)
X_effect = helmert.fit_transform(df, y)

print("Q7 – Effect Encoding (Helmert):")
print(X_effect, "\n")

# ----------------------------------------------------------------------
# Questão 8
compare = pd.DataFrame({
    "Método": ["One-Hot", "Dummy (drop-1)", "Effect / Helmert"],
    "n_colunas": [X_ohe.shape[1], X_dummy.shape[1], X_effect.shape[1]],
    "Soma_colunas = 0?": [False, False, True],
    "Correlação Perfeita": [True, False, False],
    "Interpretabilidade coef.": ["Alta", "Alta", "Média"]
})

print("=== Q8 – Tabela comparativa ===")
print(compare.to_string(index=False), "\n")

# ----------------------------------------------------------------------
# Questão 11
hasher = FeatureHasher(n_features=16, input_type="string")  # 16 buckets
rows = [[f"{col}={val}" for col, val in row.items()] for row in df.to_dict("records")]
X_hash = hasher.transform(rows).toarray()
X_hash = pd.DataFrame(X_hash, index=df.index,
                      columns=[f"hash_{i}" for i in range(X_hash.shape[1])])

print("Q11 – Feature Hashing (16 buckets): shape ->", X_hash.shape, "\n")

count_maps = {
    col: df[col].value_counts().to_dict()
    for col in df.columns
}
X_bin = df.copy()
for col, m in count_maps.items():
    X_bin[col] = X_bin[col].map(m)

print("Q12 – Bin Counting (Count Encoding):")
print(X_bin.head())
