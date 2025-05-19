import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

penguins = sns.load_dataset("penguins")

features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"]
target = "body_mass_g"
penguins = penguins[features + [target]].dropna()

X = penguins[features]
y = penguins[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

y_pred = ridge.predict(X_test_scaled)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Erro médio: {rmse:.2f} g")

coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": ridge.coef_
})
print("\nCoeficientes:")
print(coef_df.to_string(index=False))
