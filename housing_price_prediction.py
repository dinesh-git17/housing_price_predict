from logger import logger  # Import centralized logger
from data_preprocessing import load_data
from model_training import train_and_evaluate_model
from model_tuning import tune_model
from model_evaluation import evaluate_model

def main():
    logger.info("📥 Loading dataset...")
    X_train, X_test, y_train, y_test = load_data()

    logger.info("🚀 Training initial model...")
    initial_model, initial_rmse, initial_r2 = train_and_evaluate_model(X_train, X_test, y_train, y_test)

    logger.info(f"🏆 Initial Model Performance: RMSE={initial_rmse:.4f}, R²={initial_r2:.4f}")

    logger.info("🔍 Performing hyperparameter tuning...")
    best_model, best_rmse, best_r2 = tune_model(X_train, y_train)

    logger.info(f"🏆 Best Model Performance: RMSE={best_rmse:.4f}, R²={best_r2:.4f}")

    logger.info("✅ Model tuning complete! Evaluating final model...")
    evaluate_model(best_model, X_test, y_test)

if __name__ == "__main__":
    main()
