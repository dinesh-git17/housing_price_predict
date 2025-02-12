import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
from logger import logger  # Import logger

def tune_model(X_train, y_train):
    param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
    best_rmse = float("inf")
    best_model = None

    for params in ParameterGrid(param_grid):
        logger.info(f"🔍 Testing parameters: {params}")
        model = RandomForestRegressor(**params, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_train)
        rmse = mean_squared_error(y_train, y_pred) ** 0.5

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    logger.info(f"🏆 Best RMSE: {best_rmse:.4f}")
    return best_model, best_rmse, 0
