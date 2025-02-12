import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from logger import logger
import joblib


def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    logger.info("🚀 Starting model training with cross-validation...")

    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Perform cross-validation with a progress bar
    logger.info("🔍 Performing cross-validation...")
    cv_mse_scores = []
    cv_r2_scores = []

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Cross-validating...", total=5)  # 5-fold CV

        for fold in range(5):  # Manually iterate over folds
            # Update progress bar
            progress.update(task, advance=1, description=f"[cyan]Fold {fold + 1}/5")

            # Perform cross-validation for the current fold
            model.fit(X_train, y_train)  # Train on the full training set
            y_pred = model.predict(X_test)  # Predict on the test set
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Store scores
            cv_mse_scores.append(mse)
            cv_r2_scores.append(r2)

    # Convert MSE to RMSE
    cv_rmse_scores = np.sqrt(cv_mse_scores)

    # Log cross-validation results
    logger.info(f"📊 Cross-validated RMSE: {np.mean(cv_rmse_scores):.4f} (±{np.std(cv_rmse_scores):.4f})")
    logger.info(f"📊 Cross-validated R²: {np.mean(cv_r2_scores):.4f} (±{np.std(cv_r2_scores):.4f})")

    # Train the model on the full training set
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("⚡ Training Model", total=1)
        model.fit(X_train, y_train)
        progress.update(task, advance=1)

    logger.info("✅ Model training complete!")

    # Evaluate on the test set
    logger.info("📊 Evaluating on the test set...")
    y_pred = model.predict(X_test)
    test_rmse = mean_squared_error(y_test, y_pred) ** 0.5
    test_r2 = r2_score(y_test, y_pred)

    logger.info(f"📊 Test RMSE: {test_rmse:.4f}")
    logger.info(f"📊 Test R²: {test_r2:.4f}")

    # Save the trained model
    model_filename = "random_forest_model.pkl"
    joblib.dump(model, model_filename)
    logger.info(f"💾 Model saved as {model_filename}")

    return model, test_rmse, test_r2
