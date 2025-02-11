# model_tuning.py

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def tune_model(X_train, y_train):
    # Set up the RandomForestRegressor
    model = RandomForestRegressor()

    # Define hyperparameters to tune
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Get the best model and evaluation metrics
    best_model = grid_search.best_estimator_
    best_rmse = grid_search.best_score_
    best_r2 = grid_search.best_estimator_.score(X_train, y_train)

    return best_model, best_rmse, best_r2
