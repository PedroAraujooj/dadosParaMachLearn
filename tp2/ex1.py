from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

print("Dados originais (primeiras 5 linhas):")
print(df.head())

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled_data, columns=df.columns)

print("\nDados escalonados (primeiras 5 linhas):")
print(df_scaled.head())

means = df_scaled.mean()
stds = df_scaled.std(ddof=0)

print("\nMédia das features escalonadas:")
print(means)

print("\nDesvio padrão das features escalonadas:")
print(stds)