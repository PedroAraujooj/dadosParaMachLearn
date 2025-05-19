import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

var_thresh = VarianceThreshold(threshold=0.01)
X_var = var_thresh.fit_transform(X)
features_after_variance = X.columns[var_thresh.get_support()]

k = 10
kbest = SelectKBest(score_func=f_classif, k=k)
X_kbest = kbest.fit_transform(X[features_after_variance], y)
features_kbest = features_after_variance[kbest.get_support()]

corr_matrix = X[features_kbest].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

print("Features originais:", X.shape[1])
print("Após VarianceThreshold (var > 0.01):", len(features_after_variance))
print("Top 10 após o SelectKBest :", list(features_kbest))
