import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFECV
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

var_thresh = VarianceThreshold(threshold=0.01)
X_var = var_thresh.fit_transform(X)
features_after_variance = X.columns[var_thresh.get_support()]

kbest = SelectKBest(score_func=f_classif, k=10)
X_kbest = kbest.fit_transform(X[features_after_variance], y)
features_kbest = features_after_variance[kbest.get_support()]

corr_matrix = X[features_kbest].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

estimator = LogisticRegression(max_iter=5000, solver='lbfgs')
rfecv = RFECV(estimator=estimator, step=1, cv=5, scoring='accuracy')
rfecv.fit(X, y)
wrapper_features = list(X.columns[rfecv.support_])

print("Features selecionadas pela ABORDAGEM DE FILTRAGEM:")
print("  Variáveis após VarianceThreshold:", list(features_after_variance))
print("Após VarianceThreshold (var > 0.01):", len(features_after_variance))
print("Top 10 após o SelectKBest :", list(features_kbest))

print("\nFeatures selecionadas pela ABORDAGEM WRAPPER (RFECV):")
print(f"  Número de features selecionadas: {rfecv.n_features_}")
print("  Conjunto final:", wrapper_features)
