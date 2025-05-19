import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

penguins = sns.load_dataset("penguins")
X = penguins[["body_mass_g"]].dropna()

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

penguins_scaled = X.copy()
penguins_scaled["body_mass_g_scaled"] = X_scaled

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].hist(X["body_mass_g"], bins=15)
ax[0].set_title("Peso original (g)")
ax[0].set_xlabel("body_mass_g"); ax[0].set_ylabel("freq.")

ax[1].hist(penguins_scaled["body_mass_g_scaled"], bins=15)
ax[1].set_title("Peso normalizado (0 a 600)")
ax[1].set_xlabel("body_mass_g_scaled"); ax[1].set_ylabel("freq.")

plt.tight_layout()
plt.savefig("ex7.png")
