import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

penguins = sns.load_dataset("penguins")
mass_raw = penguins["body_mass_g"].dropna().values.reshape(-1, 1)

log_tf = FunctionTransformer(np.log, validate=True, feature_names_out='one-to-one')
inv_exp_tf = FunctionTransformer(lambda x: np.exp(-x), validate=True, feature_names_out='one-to-one')

pipe = Pipeline([
    ('log', log_tf),
    ('exp_inv', inv_exp_tf),
])

mass_trans = pipe.fit_transform(mass_raw).ravel()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(mass_raw.ravel(), 15)
ax[0].set_title("Peso bruto (g)")
ax[0].set_xlabel("gramas")
ax[0].set_ylabel("freq.")

ax[1].hist(mass_trans, 15)
ax[1].set_title("log ➜ exp(‑x)")
ax[1].set_xlabel("valor transformado")
ax[1].set_ylabel("freq.")

plt.tight_layout()
plt.savefig("ex10.png")
