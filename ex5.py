import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer

penguins = sns.load_dataset("penguins")
mass_raw = penguins["body_mass_g"].dropna()

log_tf = FunctionTransformer(np.log)
mass_log = log_tf.transform(mass_raw.values.reshape(-1, 1)).ravel()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].hist(mass_raw, 15)
ax[0].set_title("Peso bruto (g)")
ax[0].set_xlabel("gramas")
ax[0].set_ylabel("freq.")

ax[1].hist(mass_log, 15)
ax[1].set_title("Peso transformado – log")
ax[1].set_xlabel("log(gramas)")
ax[1].set_ylabel("freq.")

plt.tight_layout()
plt.savefig("ex5.png")
