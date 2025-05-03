import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

penguins = sns.load_dataset("penguins")
var = "body_mass_g"
data = penguins[var].dropna()

n_bins = 5
fixed_bins = pd.cut(data, bins=n_bins)
fixed_counts = fixed_bins.value_counts().sort_index()

#--------------Por quantis
quantile_bins = pd.qcut(data, q=n_bins)
quantile_counts = quantile_bins.value_counts().sort_index()

#---------------Tabela comparativa
comparison = pd.DataFrame({
    "Fixed-width bins": fixed_counts,
    "Quantile bins": quantile_counts
})
print(comparison, "\n")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ---------------- largura fixa
axes[0].hist(data, bins=n_bins)
axes[0].set_title(f"{var} – largura fixa ({n_bins})")
axes[0].set_xlabel(var)
axes[0].set_ylabel("freq.")

# -----------------quantis
edges = [iv.left for iv in quantile_bins.cat.categories] \
        + [quantile_bins.cat.categories[-1].right]
axes[1].hist(data, bins=edges)
axes[1].set_title(f"{var} – bins por quantil ({n_bins})")
axes[1].set_xlabel(var)
axes[1].set_ylabel("freq.")

plt.tight_layout()
plt.savefig("ex4.png")
