# train_lai_model.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import joblib

def generate_synthetic_data():
    np.random.seed(42)
    data = {
        "NDVI": np.random.uniform(0.2, 0.9, 100),
        "EVI": np.random.uniform(0.1, 0.8, 100),
        "GNDVI": np.random.uniform(0.2, 0.9, 100),
        "RENDVI": np.random.uniform(0.2, 0.7, 100),
        "WDRVI": np.random.uniform(0.1, 0.6, 100),
        "CI": np.random.uniform(0.3, 1.0, 100),
        "NDRE": np.random.uniform(0.1, 0.7, 100),
        "LAI": np.random.uniform(0.5, 6.0, 100)
    }
    df = pd.DataFrame(data)
    return df

# Generate synthetic data
df = generate_synthetic_data()

# Split the data into features (X) and target (y)
X = df[["NDVI", "EVI", "GNDVI", "RENDVI", "WDRVI", "CI", "NDRE"]]
y = df["LAI"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor
lai_model = RandomForestRegressor(n_estimators=100, random_state=42)
lai_model.fit(X_train, y_train)

# Evaluate the model
y_pred = lai_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")

# Save the trained model to a file
joblib.dump(lai_model, 'lai_regression_model.pkl')
print("Model saved as lai_regression_model.pkl")
