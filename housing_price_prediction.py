# housing_price_prediction.py
from data_preprocessing import preprocess_data
from model_training import train_and_evaluate_model

def main():
    # Preprocess the data
    data = preprocess_data()
    
    # Train and evaluate the model
    model, rmse, r2 = train_and_evaluate_model(data)
    
    # Print the results
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared: {r2}")

if __name__ == "__main__":
    main()
