import pandas as pd
from sklearn.datasets import load_iris


def manual_standard_scaler(df: pd.DataFrame):

    means = df.mean()
    stds = df.std(ddof=0)
    df_scaled = (df - means) / stds
    return df_scaled


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

df_scaled = manual_standard_scaler(df)


print("Primeiras 5 linhas dos dados originais :")
print(df.head())

print("Primeiras 5 linhas dos dados escalonados manualmente:")
print(df_scaled.head())

print("\nVerificação: média das features escalonadas (deve ser ~0):")
print(df_scaled.mean())

print("\nVerificação: desvio padrão das features escalonadas (deve ser 1):")
print(df_scaled.std(ddof=0))
