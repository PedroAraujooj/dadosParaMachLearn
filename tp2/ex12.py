import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")
nltk.download("omw-1.4")

texto = (
    "A alma é, pois, imortal; renasceu repetidas vezes na existência e contemplou "
    "todas as coisas existentes e por isso não há nada que ela não conheça! Não é "
    "de espantar que ela seja capaz de evocar à memória a lembrança de objetos que "
    "viu anteriormente, e que se relacionam tanto com a virtude como com as outras "
    "coisas existentes. Toda a natureza, com efeito, é uma só, é um todo orgânico, "
    "e o espírito já viu todas as coisas; logo, nada impede que ao nos lembrarmos "
    "de uma coisa – o que nós, homens, chamamos de aprendizado – descubramos de "
    "novo aquilo que já sabíamos!"
)

tokens = wordpunct_tokenize(texto.lower())

lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(tok) for tok in tokens if tok.isalpha()]

texto_lemmatizado = " ".join(lemmas)

print("Tokens lematizados:")
print(lemmas)
print("\nTexto reconstruído:")
print(texto_lemmatizado)
