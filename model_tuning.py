from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
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
            scoring="neg_mean_squared_error",
            n_jobs=-1,  # Use all available CPU cores
            verbose=0
        )

        # Fit the grid search
        grid_search.fit(X_train, y_train)

    # Log the best parameters and score
    logger.info(f"🏆 Best Parameters: {grid_search.best_params_}")
    logger.info(f"🏆 Best Cross-validated RMSE: {(-grid_search.best_score_) ** 0.5:.4f}")

    # Return the best model and its performance metrics
    best_model = grid_search.best_estimator_
    return best_model, (-grid_search.best_score_) ** 0.5, 0
