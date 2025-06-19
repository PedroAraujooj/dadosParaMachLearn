import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from scipy import sparse

file_path = Path(__file__).with_name("openpowerlifting.csv")
df = pd.read_csv(file_path)

obj_cols = df.select_dtypes(include="object").columns.tolist()
categorical_cols = sorted(set(obj_cols))

print(f"CATEGORICAL COLS ({len(categorical_cols)}): {categorical_cols}")

X_num = df.drop(columns=categorical_cols)
X_cat = df[categorical_cols]

#  ONE-HOT ==============================================
ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
X_ohe = ohe.fit_transform(X_cat)

print("\n[A] ONE-HOT ENCODING")
print(f"Shape: {X_ohe.shape}  –  densidade: {X_ohe.nnz / np.prod(X_ohe.shape):.4f}")
print("Primeiras 5 colunas:", ohe.get_feature_names_out()[:5])

# DUMMY =============================================================================
dummy = OneHotEncoder(sparse_output=True, drop="first", handle_unknown="ignore")
X_dummy = dummy.fit_transform(X_cat)

print("\n[B] DUMMY CODING (drop first)")
print(f"Shape: {X_dummy.shape}  –  densidade: {X_dummy.nnz / np.prod(X_dummy.shape):.4f}")
print("Primeiras 5 colunas:", dummy.get_feature_names_out()[:5])


#  EFFECT ======================================
def effect_coding(df_cat: pd.DataFrame, max_levels=None):
    mats, names = [], []

    for col in df_cat.columns:
        n_levels = df_cat[col].nunique(dropna=False)
        if max_levels and n_levels > max_levels:
            continue

        dummies = pd.get_dummies(df_cat[col], dtype=np.int8, sparse=True)
        levels = dummies.columns.tolist()
        ref = levels[-1]
        eff = dummies.iloc[:, :-1].sparse.to_coo().tolil()
        ref_idx = np.where(df_cat[col].values == ref)[0]
        if ref_idx.size:
            eff[ref_idx, :] = -1
        mats.append(eff.tocsr())
        names.extend([f"{col}[{lvl}]_eff" for lvl in levels[:-1]])

    return sparse.hstack(mats, format="csr"), names


X_eff, eff_names = effect_coding(X_cat)

print("\n[C] EFFECT CODING")
print(f"Shape: {X_eff.shape}  –  densidade: {X_eff.nnz / np.prod(X_eff.shape):.4f}")
print("Primeiras 5 colunas:", eff_names[:5])

# HASHING =======================
tokens = X_cat.astype(str).agg(lambda row: [f"{c}={v}" for c, v in row.items()], axis=1)
hasher = FeatureHasher(n_features=2 ** 12, input_type="string")
X_hash = hasher.transform(tokens)

print("\n[i] FEATURE HASHING")
print(f"Shape: {X_hash.shape}  –  densidade: {X_hash.nnz / np.prod(X_hash.shape):.4f}")

# Bin counting ==============
X_bin = X_hash.copy()
X_bin.data = np.abs(X_bin.data)
print("\n[ii] BIN COUNTING")
print(f"Shape: {X_bin.shape}  –  densidade: {X_bin.nnz / np.prod(X_bin.shape):.4f}")
