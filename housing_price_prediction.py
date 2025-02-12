from logger import logger  # Import centralized logger
from data_preprocessing import load_data
from model_training import train_and_evaluate_model
from model_tuning import tune_model
from model_evaluation import evaluate_model
import joblib


def main():
    try:
        # Step 1: Load data
        logger.info("📥 Loading dataset...")
        X_train, X_test, y_train, y_test = load_data()

        # Step 2: Train and evaluate initial model
        logger.info("🚀 Training initial model...")
        initial_model, initial_rmse, initial_r2 = train_and_evaluate_model(X_train, X_test, y_train, y_test)
        logger.info(f"🏆 Initial Model Performance: RMSE={initial_rmse:.4f}, R²={initial_r2:.4f}")

        # Step 3: Perform hyperparameter tuning
        logger.info("🔍 Performing hyperparameter tuning...")
        best_model, best_rmse, best_r2 = tune_model(X_train, y_train)
        logger.info(f"🏆 Best Model Performance: RMSE={best_rmse:.4f}, R²={best_r2:.4f}")

        # Step 4: Save the best model
        best_model_filename = "best_random_forest_model.pkl"
        joblib.dump(best_model, best_model_filename)
        logger.info(f"💾 Best model saved as {best_model_filename}")

        # Step 5: Evaluate the best model on the test set
        logger.info("✅ Model tuning complete! Evaluating final model...")
        evaluate_model(best_model, X_test, y_test)

    except Exception as e:
        logger.error(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    main()
