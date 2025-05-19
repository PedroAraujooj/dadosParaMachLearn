from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt

text = """
A alma é, pois, imortal; renasceu repetidas vezes na existência e contemplou todas as coisas existentes 
e por isso não há nada que ela não conheça! Não é de espantar que ela seja capaz de evocar à memória 
a lembrança de objetos que viu anteriormente, e que se relacionam tanto com a virtude como com as outras 
coisas existentes. Toda a natureza, com efeito, é uma só, é um todo orgânico, e o espírito já viu todas 
as coisas; logo, nada impede que ao nos lembrarmos de uma coisa – o que nós, homens, chamamos de “saber” 
– todas as outras coisas acorram imediata e maquinalmente à nossa consciência.
"""

vectorizer = CountVectorizer()
X = vectorizer.fit_transform([text])

feature_names = vectorizer.get_feature_names_out()
counts = X.toarray()[0]
df_bow = pd.DataFrame({
    'term': feature_names,
    'count': counts
})

df_sorted = df_bow.sort_values(by='count', ascending=False).reset_index(drop=True)

plt.figure(figsize=(12, 6))
plt.bar(df_sorted['term'], df_sorted['count'])
plt.xticks(rotation=90, ha='right')
plt.xlabel('Termo')
plt.ylabel('Contagem')
plt.title('Frequência de todas as palavras (Bag-of-Words)')
plt.tight_layout()
plt.show()
