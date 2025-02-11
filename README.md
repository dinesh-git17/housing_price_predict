# Housing Price Prediction

This project involves building a machine learning model to predict housing prices based on various features such as median income, house age, number of rooms, and other related features. The workflow includes data preprocessing, model training, evaluation, and optimization.

## Project Structure

- **`housing_price_prediction.py`**: Main script that runs the entire pipeline—loads data, preprocesses it, trains the model, evaluates it, and tunes the model's hyperparameters.
- **`data_preprocessing.py`**: Handles the data cleaning, feature extraction, and splits the data into training and testing sets.
- **`model_training.py`**: Trains the model (currently using a Random Forest Regressor), evaluates it, and calculates performance metrics (RMSE and R-squared).
- **`model_tuning.py`**: Fine-tunes the model’s hyperparameters using techniques like Grid Search for optimal performance.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn

## How It Works

### Data Preprocessing (`data_preprocessing.py`):
- Loads and cleans the dataset.
- Extracts features and target labels from the data.
- Splits the data into training and testing sets.

### Model Training and Evaluation (`model_training.py`):
- Trains a `RandomForestRegressor` model using the preprocessed data.
- Evaluates the model's performance on the test data.
- Calculates RMSE (Root Mean Squared Error) and R-squared.

### Model Tuning (`model_tuning.py`):
- Uses Grid Search to tune hyperparameters for the model.
- Finds the best configuration for improved model performance.

### Main Script (`housing_price_prediction.py`):
- Calls the preprocessing function to prepare the data.
- Trains and evaluates the model.
- Optionally tunes the model's hyperparameters to further improve performance.

## Output

- **RMSE (Root Mean Squared Error):** Indicates how far the model’s predictions are from the actual values. Lower values are better.
- **R-squared:** Indicates how well the model fits the data. Higher values are better, with 1 indicating a perfect fit.
