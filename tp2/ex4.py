import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import Normalizer
import numpy as np

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

normalizer = Normalizer(norm='l1')
scaled_data = normalizer.fit_transform(df)
df_l1 = pd.DataFrame(scaled_data, columns=df.columns)

print("Dados originais (primeiras 5 linhas):")
print(df.head())

print("Primeiras 5 linhas dos dados após regularização L1:")
print(df_l1.head())

sums = np.sum(np.abs(df_l1.values), axis=1)
print("\nSomatório L1 das primeiras 5 amostras (deve ser 1):")
print(sums[:5])