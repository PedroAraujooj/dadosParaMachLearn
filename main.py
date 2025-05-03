import seaborn as sns
import matplotlib.pyplot as plt

# Carrega o dataset
penguins = sns.load_dataset("penguins")

# Seleciona apenas colunas numéricas
num_cols = penguins.select_dtypes(include="number").columns

# Define layout da figura
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
axes = axes.flatten()

# Plota um histograma para cada variável
for i, col in enumerate(num_cols):
    axes[i].hist(penguins[col].dropna(), bins=15)
    axes[i].set_title(col)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequência")

plt.tight_layout()
plt.savefig("main.png")
