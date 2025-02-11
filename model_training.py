# model_training.py

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np  # For sqrt function

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    # Create the model
    model = RandomForestRegressor()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # Take the square root of MSE to get RMSE
    r2 = r2_score(y_test, y_pred)

    return model, rmse, r2
