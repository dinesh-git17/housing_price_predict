# Housing Price Prediction

This project is a machine learning pipeline for predicting housing prices using the **California Housing dataset** from `sklearn.datasets`. The pipeline includes data loading, preprocessing, model training, hyperparameter tuning, evaluation, and logging.

---

## 📌 Project Structure

```
housing_price_predict/
│── logs/                     # Stores log files
│── housing_price_prediction.py  # Main script
│── data_processing.py         # Data preprocessing
│── model_training.py          # Model training
│── model_tuning.py            # Hyperparameter tuning
│── model_evaluation.py        # Model evaluation
│── logger.py                  # Logging setup
│── README.md                  # Project documentation
```

---

## 🚀 Setup Instructions

### 1️⃣ Install Dependencies

Ensure you have Python installed, then run:

```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Pipeline

```bash
python housing_price_prediction.py
```

---

## 📜 Scripts Overview

### `housing_price_prediction.py` (Main Script)
- The entry point for the project.
- Calls all other scripts in the correct sequence.
- Loads and preprocesses data.
- Trains an initial model and evaluates it.
- Performs hyperparameter tuning to improve the model.
- Runs final evaluation and logs results.

---

### `data_processing.py` (Data Preprocessing)
- Loads the **California Housing dataset**.
- Splits data into training and testing sets.
- Normalizes features using **StandardScaler**.
- Logs progress during data loading and preprocessing.

---

### `model_training.py` (Model Training)
- Initializes a **RandomForestRegressor** model.
- Uses a **progress bar** to visualize training.
- Fits the model to the training data.
- Generates predictions and evaluates performance using **RMSE and R² score**.
- Logs key training results.

---

### `model_tuning.py` (Hyperparameter Tuning)
- Uses a **grid search** approach to test different hyperparameters.
- Trains multiple **RandomForestRegressor** models with varying settings.
- Identifies the best model based on **RMSE and R² score**.
- Logs the best model configuration.

---

### `model_evaluation.py` (Model Evaluation)
- Uses the best model found in tuning.
- Makes final predictions on test data.
- Calculates **RMSE and R² score** for model performance.
- Logs final evaluation results.

---

### `logger.py` (Logging Setup)
- Initializes and configures a **color-coded logger**.
- Ensures logs are formatted with timestamps and severity levels.
- Prevents duplicate logs when importing the logger into multiple scripts.

---

## 🏆 Features

✔ Uses **RandomForestRegressor** for predictions  
✔ **Hyperparameter tuning** using `ParameterGrid`  
✔ **Rich progress bars** for model training visualization  
✔ **Colorful logging** with `colorlog`  
✔ **Clean modular structure**  

---

## 📌 Requirements

- Python 3.11+
- Libraries: `scikit-learn`, `pandas`, `rich`, `colorlog`

---

## 📢 Future Improvements

- Add **feature selection** for better performance.
- Implement **cross-validation** for robust tuning.
- Explore **other models** like Gradient Boosting.
- Deploy model as a **REST API**.

---

## 📬 Contact

For any issues or improvements, feel free to contribute or reach out.

---
