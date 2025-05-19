from sklearn.feature_extraction.text import CountVectorizer

text = """
A alma é, pois, imortal; renasceu repetidas vezes na existência e contemplou todas as coisas existentes 
e por isso não há nada que ela não conheça! Não é de espantar que ela seja capaz de evocar à memória 
a lembrança de objetos que viu anteriormente, e que se relacionam tanto com a virtude como com as outras 
coisas existentes. Toda a natureza, com efeito, é uma só, é um todo orgânico, e o espírito já viu todas 
as coisas; logo, nada impede que ao nos lembrarmos de uma coisa – o que nós, homens, chamamos de “saber” 
– todas as outras coisas acorram imediata e maquinalmente à nossa consciência.
"""

vectorizer1 = CountVectorizer(ngram_range=(1, 1))
X1 = vectorizer1.fit_transform([text])
n_unigrams = len(vectorizer1.get_feature_names_out())

vectorizer2 = CountVectorizer(ngram_range=(2, 2))
X2 = vectorizer2.fit_transform([text])
n_bigrams = len(vectorizer2.get_feature_names_out())

vectorizer3 = CountVectorizer(ngram_range=(3, 3))
X3 = vectorizer3.fit_transform([text])
n_trigrams = len(vectorizer3.get_feature_names_out())

print(f"Número de unigrams: {n_unigrams}")
print(f"Número de bigrams: {n_bigrams}")
print(f"Número de trigrams: {n_trigrams}")
