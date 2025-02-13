from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from logger import logger
from rich.progress import Progress, SpinnerColumn, TextColumn


def tune_model(X_train, y_train):
    # Display a visually appealing message with a spinner
    with Progress(
        SpinnerColumn(),  # Spinner animation
        TextColumn("[bold cyan]{task.description}"),  # Text description
        transient=True  # Hide the progress bar after completion
    ) as progress:
        task = progress.add_task(
            "🔍 Starting hyperparameter tuning with cross-validation...",
            total=None  # Indeterminate progress
        )

        # Define the parameter grid
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [10, 20, None],
            "max_features": ["sqrt", "log2"]
        }

        # Initialize the model
        model = RandomForestRegressor(random_state=42)

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,  # 5-fold cross-validation
            scoring="neg_mean_squared_error",  # Use negative MSE for scoring
            n_jobs=-1,  # Use all available CPU cores
            verbose=0
        )

        # Fit the grid search
        grid_search.fit(X_train, y_train)

    # Log the best parameters and validation set performance
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_rmse = (-grid_search.best_score_) ** 0.5  # RMSE on validation set

    # Compute R² on the validation set
    y_pred = best_model.predict(X_train)  # Predict on the training set (for validation)
    best_r2 = r2_score(y_train, y_pred)

    logger.info(f"🏆 Best Parameters: {best_params}")
    logger.info(f"🏆 Best Cross-validated RMSE: {best_rmse:.4f}")
    logger.info(f"🏆 Best Cross-validated R²: {best_r2:.4f}")

    # Return the best model and its performance metrics
    return best_model, best_rmse, best_r2
