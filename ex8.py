import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

# 1. Carrega o dataset e seleciona a variável
penguins = sns.load_dataset("penguins")
X = penguins[["body_mass_g"]].dropna()

# 2. Aplica Min-Max Scaling para o intervalo [0, 600]
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# 3. Cria DataFrame com o resultado
penguins_scaled = X.copy()
penguins_scaled["body_mass_g_scaled"] = X_scaled

# 4. Plota os histogramas
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].hist(X["body_mass_g"], bins=15)
ax[0].set_title("Peso original (g)")
ax[0].set_xlabel("body_mass_g"); ax[0].set_ylabel("freq.")

ax[1].hist(penguins_scaled["body_mass_g_scaled"], bins=15)
ax[1].set_title("Peso normalizado (0 a 600)")
ax[1].set_xlabel("body_mass_g_scaled"); ax[1].set_ylabel("freq.")

plt.tight_layout()
plt.savefig("ex7.png")
