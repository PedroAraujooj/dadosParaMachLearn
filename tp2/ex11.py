from nltk.tokenize import wordpunct_tokenize
from nltk.stem.snowball import SnowballStemmer

text = """
A alma é, pois, imortal; renasceu repetidas vezes na existência e contemplou todas as coisas existentes 
e por isso não há nada que ela não conheça! Não é de espantar que ela seja capaz de evocar à memória 
a lembrança de objetos que viu anteriormente, e que se relacionam tanto com a virtude como com as outras 
coisas existentes. Toda a natureza, com efeito, é uma só, é um todo orgânico, e o espírito já viu todas 
as coisas; logo, nada impede que ao nos lembrarmos de uma coisa – o que nós, homens, chamamos de “saber” 
– todas as outras coisas acorram imediata e maquinalmente à nossa consciência.
"""

stemmer = SnowballStemmer("portuguese")

tokens = wordpunct_tokenize(text.lower())
stemmed_tokens = [stemmer.stem(tok) for tok in tokens if tok.isalpha()]

stemmed_text = " ".join(stemmed_tokens)

print("Tokens após stemming:\n")
print(stemmed_tokens)
print("\nTexto reconstruído após stemming:\n")
print(stemmed_text)
