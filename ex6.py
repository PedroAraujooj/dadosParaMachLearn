import seaborn as sns
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt

penguins = sns.load_dataset("penguins")
X = penguins[["flipper_length_mm"]].dropna()

pt = PowerTransformer()
X_pt = pt.fit_transform(X)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(X.values.ravel(), bins=15)
ax[0].set_title("flipper_length_mm (bruto)")

ax[1].hist(X_pt.ravel(), bins=15)
ax[1].set_title("flipper_length_mm (PowerTransform)")
plt.tight_layout()
plt.savefig("ex6.png")
