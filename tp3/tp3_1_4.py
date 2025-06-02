import html
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ---------- Questão 1: carregar dados + TF-IDF ----------
ROOT = Path(__file__).parent / "mdb"
TRAIN_DIR = ROOT / "train"
TEST_DIR = ROOT / "test"


def clean_html(text):
    text = html.unescape(text)
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    return re.sub(r"\s+", " ", text).strip()


def read_split(split_dir: Path):
    texts, labels = [], []
    for label_dir, y in [("pos", 1), ("neg", 0)]:
        for txt_file in (split_dir / label_dir).glob("*.txt"):
            texts.append(clean_html(txt_file.read_text(encoding="utf-8", errors="ignore")))
            labels.append(y)
    return pd.DataFrame({"text": texts, "y": labels})


print("Inicio leitura dos dados de treino")
df_train = read_split(TRAIN_DIR)
print("Inicio leitura dos dados de teste")
df_test = read_split(TEST_DIR)

# Vetorização TF-IDF (remoção básica de stop-words em inglês)
print("Inicio vetorização")
tfidf = TfidfVectorizer(max_df=0.8,
                        min_df=5,
                        stop_words="english",
                        ngram_range=(1, 2),
                        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b")
print("Tranformação e treinamento dos dados de treino")
X_train = tfidf.fit_transform(df_train["text"])
print("Tranformação e treinamento dos dados de teste")
X_test = tfidf.transform(df_test["text"])

# ---------- Questão 2: 10 maiores e 10 menores TF-IDF ----------
print("Inicio da classificação")
mean_tfidf = np.asarray(X_train.mean(axis=0)).ravel()
feat_names = np.array(tfidf.get_feature_names_out())

top10_idx = mean_tfidf.argsort()[-10:]  # 10 maiores
bot10_idx = mean_tfidf.argsort()[:10]  # 10 menores

print("=== 10 features com MAIOR TF-IDF médio ===")
for f, v in zip(feat_names[top10_idx][::-1], mean_tfidf[top10_idx][::-1]):
    print(f"{f:<25} {v:.4f}")

print("\n=== 10 features com MENOR TF-IDF médio ===")
for f, v in zip(feat_names[bot10_idx], mean_tfidf[bot10_idx]):
    print(f"{f:<25} {v:.4f}")


# ---------- Questão 3: Regressão Logística ----------
logreg = LogisticRegression(max_iter=400, n_jobs=-1)
logreg.fit(X_train, df_train["y"])

y_pred = logreg.predict(X_test)
acc = accuracy_score(df_test["y"], y_pred)
print(f"Precisão no conjunto de teste: {acc:.4%}\n")

print("Relatório de classificação:")
print(classification_report(df_test["y"], y_pred, target_names=["neg", "pos"]))

# ---------- Questão 4: Top 40 coeficientes ----------
coef = logreg.coef_.ravel()
top_pos_idx = coef.argsort()[-40:]
top_neg_idx = coef.argsort()[:40]


def plot_bar(indices, title):
    words = feat_names[indices]
    vals = coef[indices]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), vals, tick_label=words)
    plt.title(title)
    plt.xlabel("Coeficiente da Regressão Logística")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


plot_bar(top_pos_idx, "40 Maiores Coeficientes (Preditores de Sentimento POSITIVO)")
plot_bar(top_neg_idx, "40 Menores Coeficientes (Preditores de Sentimento NEGATIVO)")
