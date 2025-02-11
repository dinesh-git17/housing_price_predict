import ssl
import certifi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 🔹 Fix SSL certificate issues
ssl_context = ssl.create_default_context()
ssl_context.load_verify_locations(certifi.where())
ssl._create_default_https_context = ssl.create_default_context

# 🔹 Load the dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

# 🔹 Show basic info
print(df.head())
print(df.describe())

# 🔹 Split dataset into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=["Target"]), df["Target"], test_size=0.2, random_state=42
)

# 🔹 Scale features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔹 Train a Linear Regression Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 🔹 Make predictions
y_pred = model.predict(X_test_scaled)

# 🔹 Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared Score (R²): {r2:.4f}")

# 🔹 Plot Actual vs. Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "--", color="red")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Housing Prices")
plt.show()
