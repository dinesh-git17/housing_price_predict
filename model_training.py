import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from logger import logger  # Import logger

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    logger.info("🚀 Starting model training...")

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("⚡ Training Model", total=1)
        model.fit(X_train, y_train)
        progress.update(task, advance=1)

    logger.info("✅ Model training complete!")

    logger.info("📊 Generating predictions...")
    y_pred = model.predict(X_test)

    logger.info("📈 Evaluating model performance...")
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    logger.info(f"🔢 RMSE: {rmse:.4f}")
    logger.info(f"📊 R² Score: {r2:.4f}")

    return model, rmse, r2
