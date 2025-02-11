# housing_price_prediction.py

from data_preprocessing import preprocess_data
from model_training import train_and_evaluate_model
from model_tuning import tune_model

def main():
    # Step 1: Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data()

    # Step 2: Train and evaluate the initial model
    model, rmse, r2 = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    print(f"Initial Model RMSE: {rmse}")
    print(f"Initial Model R-squared: {r2}")

    # Step 3: Tune the model
    best_model, best_rmse, best_r2 = tune_model(X_train, y_train)
    print(f"Best Model RMSE: {best_rmse}")
    print(f"Best Model R-squared: {best_r2}")

if __name__ == "__main__":
    main()
