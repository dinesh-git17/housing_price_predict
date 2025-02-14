Housing Price Prediction Project
================================

Overview:
---------
This project is a machine learning pipeline for predicting housing prices using the California Housing dataset from scikit-learn. The pipeline includes data loading, model training, hyperparameter tuning, evaluation, logging, and model persistence. An interactive CLI is provided for making predictions, and you can choose to use a saved model or retrain a new one.

Project Structure:
------------------
housing_price_predict/
    logs/                         - Stores log files.
    housing_price_prediction.py   - Main script that manages the full pipeline.
    data_preprocessing.py         - Loads and preprocesses the California Housing dataset.
    model_training.py             - Trains a RandomForestRegressor and evaluates its performance.
    model_tuning.py               - Performs hyperparameter tuning using GridSearchCV.
    model_evaluation.py           - Evaluates the final model using RMSE and R² score.
    model_loader.py               - Loads a pre-trained model from disk.
    logger.py                     - Sets up a color-coded logger.
    predict.py                    - Interactive CLI for making predictions.
    README.txt                    - This project documentation.

Setup Instructions:
-------------------
1. Install Dependencies:
   Ensure you have Python 3.11+ installed, then run:
       pip install -r requirements.txt

2. Run the Pipeline:
   Execute the main script:
       python housing_price_prediction.py
   The script will:
       - Load and preprocess the data.
       - Check for a saved model and prompt whether to use it or retrain a new model.
       - Train, tune, and evaluate the model as needed.
       - Optionally allow interactive predictions.

3. Making Predictions:
   After the pipeline completes, you will be prompted to make predictions using the trained model.
   For a single prediction, provide 8 feature values in the following order:
       1. MedInc     - Median income in the block group
       2. HouseAge   - Median house age in the block group
       3. AveRooms   - Average number of rooms per household
       4. AveBedrms  - Average number of bedrooms per household
       5. Population - Population of the block group
       6. AveOccup   - Average occupancy (household size)
       7. Latitude   - Latitude coordinate
       8. Longitude  - Longitude coordinate
   Alternatively, you can supply a CSV file with these columns in the header:
       MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude
   The interactive CLI (in predict.py) will guide you through the input process.

Features:
---------
- Uses the California Housing dataset from scikit-learn.
- Modular pipeline with data preprocessing, training, tuning, evaluation, and logging.
- Implements RandomForestRegressor for house price prediction.
- Hyperparameter tuning using GridSearchCV.
- Color-coded logging with the colorlog library.
- Interactive CLI for predictions with clear instructions.
- Option to use a saved model or retrain a new one.

Future Improvements:
--------------------
- Deploy the model as a REST API using frameworks like Flask or FastAPI.
- Containerize the application with Docker.
- Add model explainability using SHAP or LIME.
- Develop a user-friendly web interface.
- Integrate CI/CD for automated testing and deployment.

Contact:
--------
For questions, issues, or contributions, please open an issue or submit a pull request.

Thank you for exploring the Housing Price Prediction project!
