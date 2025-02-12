import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from logger import logger  # Import logger

def evaluate_model(model, X_test, y_test):
    logger.info("🔍 Evaluating the model...")

    if model is None:
        logger.error("❌ Error: Model is None. Make sure it's trained before evaluation.")
        return

    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        logger.error(f"❌ Error while predicting: {e}")
        return

    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    logger.info(f"📊 Final RMSE: {rmse:.4f}")
    logger.info(f"📊 Final R² Score: {r2:.4f}")
