from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def tune_model(X_train, y_train):
    # Define the model
    model = RandomForestRegressor()

    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    # Set up GridSearchCV with a progress bar
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)

    # Wrap the fit method with tqdm to show a progress bar
    # Use a tqdm loop to track the progress of the grid search
    tqdm(grid_search.fit(X_train, y_train), desc="Grid Search Progress", total=grid_search.get_n_splits(), position=0)

    return grid_search.best_estimator_
