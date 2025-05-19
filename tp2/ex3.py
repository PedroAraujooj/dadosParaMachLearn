import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import Normalizer
import numpy as np

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

print("Dados originais (primeiras 5 linhas):")
print(df.head())

normalizer = Normalizer(norm='l2')
scaled_data = normalizer.fit_transform(df)
df_l2 = pd.DataFrame(scaled_data, columns=df.columns)

print("\nDados após normalização L2 (primeiras 5 linhas):")
print(df_l2.head())

norms = np.linalg.norm(df_l2.values, axis=1)
print("\nNormas L2 das primeiras 5 amostras (deve ser 1):")
print(norms[:5])