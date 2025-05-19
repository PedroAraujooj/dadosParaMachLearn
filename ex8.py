import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

penguins = sns.load_dataset("penguins")
cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
X = penguins[cols].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scaled_df = pd.DataFrame(X_scaled, columns=[col + "_zscore" for col in cols])

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(X["body_mass_g"], bins=15)
ax[0].set_title("Peso original (g)")

ax[1].hist(scaled_df["body_mass_g_zscore"], bins=15)
ax[1].set_title("Peso padronizado (StandardScaler)")
plt.tight_layout()
plt.savefig("ex8.png")
