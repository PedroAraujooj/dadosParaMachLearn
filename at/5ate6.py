import html
import re
import string
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from nltk import word_tokenize, download
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).parent.parent / "tp3" / "mdb"
TRAIN_DIR = ROOT / "train"
TEST_DIR = ROOT / "test"

for pkg in ("punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    try:
        download(pkg, quiet=True)
    except:
        pass


def clean_html(text: str) -> str:
    text = html.unescape(text)
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_split(split_dir: Path) -> pd.DataFrame:
    texts, labels = [], []
    for label_dir, y in [("pos", 1), ("neg", 0)]:
        for txt_file in (split_dir / label_dir).glob("*.txt"):
            texts.append(clean_html(txt_file.read_text(encoding="utf-8", errors="ignore")))
            labels.append(y)
    return pd.DataFrame({"text": texts, "y": labels})


#   Leitura dados =====================
print("Início leitura dos dados de treino")
df_train = read_split(TRAIN_DIR)
print("Início leitura dos dados de teste")
df_test = read_split(TEST_DIR)

# 5-A stopwords ====================================
STOPWORDS = set(stopwords.words("english"))


def remove_stopwords(doc: str) -> str:
    tokens = word_tokenize(doc.lower())
    return " ".join(t for t in tokens if t not in STOPWORDS and t not in string.punctuation)


df_train["no_stop"] = df_train["text"].apply(remove_stopwords)
df_test["no_stop"] = df_test["text"].apply(remove_stopwords)
# 5-B  Stemming ==============================================
STEMMER = PorterStemmer()


def stem_text(doc: str) -> str:
    return " ".join(STEMMER.stem(t) for t in word_tokenize(doc))


df_train["stem"] = df_train["no_stop"].apply(stem_text)
df_test["stem"] = df_test["no_stop"].apply(stem_text)

# 5-C  Lemmatization --------------------------------------
LEMMATZR = WordNetLemmatizer()


def lemmatize_text(doc: str) -> str:
    return " ".join(LEMMATZR.lemmatize(t) for t in word_tokenize(doc))


df_train["lemma"] = df_train["no_stop"].apply(lemmatize_text)
df_test["lemma"] = df_test["no_stop"].apply(lemmatize_text)

# 5-D  Bag-of-Words  ===========================
bow_vec = CountVectorizer()
X_train_bow = bow_vec.fit_transform(df_train["stem"])
X_test_bow = bow_vec.transform(df_test["stem"])
print(f"[5-D] Bag-of-Words (stemming) — shape treino: {X_train_bow.shape}, teste: {X_test_bow.shape}")

#  5-E  Bag-of-Bigrams -=====================================
bigram_vec = CountVectorizer(ngram_range=(2, 2), min_df=3)
X_train_bigram = bigram_vec.fit_transform(df_train["lemma"])
X_test_bigram = bigram_vec.transform(df_test["lemma"])
print(f"[5-E] Bag-of-Bigrams (lemmatização) — shape treino: {X_train_bigram.shape}, teste: {X_test_bigram.shape}")

# 6-A  TF-IDF ======================================================
tfidf_vec = TfidfVectorizer(max_df=0.95, min_df=2, sublinear_tf=True)
X_train_tfidf = tfidf_vec.fit_transform(df_train["no_stop"])
X_test_tfidf = tfidf_vec.transform(df_test["no_stop"])
print(f"[6-A] TF-IDF — shape treino: {X_train_tfidf.shape}, teste: {X_test_tfidf.shape}")

#  6-B  Regressão logística ===================================
clf = LogisticRegression(max_iter=400, solver="liblinear")
clf.fit(X_train_tfidf, df_train["y"])
y_pred = clf.predict(X_test_tfidf)

print("\n[6-B] Acurácia no conjunto de teste:",
      f"{accuracy_score(df_test['y'], y_pred):.4f}")

print("\nRelatório de classificação:")
print(classification_report(df_test["y"], y_pred, target_names=["neg", "pos"]))

print("Matriz de confusão:\n", confusion_matrix(df_test["y"], y_pred))
