# model_training.py
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

def train_and_evaluate_model(data):
    # Unpack the data
    X_train, X_test, y_train, y_test = data
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Compute RMSE manually
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Compute R-squared score
    r2 = model.score(X_test, y_test)

    return model, rmse, r2
