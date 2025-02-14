Housing Price Prediction Project - README.txt
=============================================

Overview:
---------
This project is a machine learning pipeline for predicting housing prices using the California Housing dataset from scikit-learn. It includes data loading, model training, hyperparameter tuning, evaluation, and logging. In addition, the project offers several visualization tools and interactive interfaces for making predictions.

Key Components:
---------------
1. Data Preprocessing:
   - Loads and preprocesses the California Housing dataset.
   - Splits the data into training and testing sets.
   
2. Model Training & Tuning:
   - Trains a RandomForestRegressor model.
   - Performs hyperparameter tuning using GridSearchCV.
   - Evaluates the model using RMSE and R² metrics.
   - Saves the trained model for future use.

3. Interactive Prediction:
   - Provides a CLI (via predict.py) to make predictions on new input.
   - Offers detailed instructions on required feature inputs.
   
4. Visualizations:
   - Visualize data distributions (histograms, scatter plots) with visualize.py.
   - Generate geospatial maps of the data using Folium in geoviz.py.
   
5. Web Dashboard:
   - An interactive Streamlit dashboard (app.py) for real-time predictions.
   - Descriptive input fields with explanations.
   - Includes an option to exit the dashboard and return to the terminal.

6. Main Interface:
   - The main script (housing_price_prediction.py) presents a unified menu with options:
     1. Run the full pipeline (train, evaluate, and predict).
     2. Visualize data (histograms & scatter plots).
     3. Generate a geospatial map.
     4. Launch the interactive Streamlit dashboard.
     5. Exit the program (with an option to return to the menu).

Project Structure:
------------------
```
housing_price_predict/
├── logs/                         - Log files.
├── housing_price_prediction.py   - Main script with the unified menu.
├── data_preprocessing.py         - Data loading and preprocessing.
├── model_training.py             - Model training using RandomForestRegressor.
├── model_tuning.py               - Hyperparameter tuning using GridSearchCV.
├── model_evaluation.py           - Model evaluation (RMSE, R²).
├── model_loader.py               - Loads a saved model.
├── logger.py                     - Logging configuration.
├── predict.py                    - Interactive CLI for predictions.
├── visualize.py                  - Data visualizations (histograms, scatter plots).
├── geoviz.py                     - Geospatial map generation using Folium.
├── app.py                        - Streamlit dashboard for interactive predictions.
└── README.txt                    - This documentation file.
```
Setup Instructions:
-------------------
1. Install Dependencies:
   Ensure you have Python 3.11+ installed, then run:
       pip install -r requirements.txt

2. Run the Main Pipeline:
   Execute the main script from the terminal:
       python housing_price_prediction.py
   You will see a menu with options to run the pipeline, visualize data, generate maps, launch the dashboard, or exit.

3. Streamlit Dashboard:
   To launch the interactive dashboard, choose option 4 from the main menu, or run:
       streamlit run app.py
   Use the dashboard to enter detailed housing block features and obtain real-time predictions.
   An "Exit Dashboard" button allows you to close the dashboard and return to the terminal.

4. Making Predictions:
   After training or loading a saved model, you can:
     - Use the CLI (predict.py) to enter 8 feature values (descriptive labels are provided in the dashboard).
     - Or, supply a CSV file with the required columns for batch predictions.

How It Works:
-------------
- The main script loads the data and checks for a saved model.
- You are given the option to use the saved model or retrain a new one.
- The training process includes cross-validation and hyperparameter tuning.
- The trained model is saved to disk for future predictions.
- Visualization tools help you explore the data and understand model performance.
- The Streamlit dashboard provides a user-friendly web interface for making predictions.

Future Enhancements:
--------------------
- Deploy the model as a REST API (using Flask, FastAPI, etc.).
- Containerize the application with Docker.
- Integrate model explainability tools (e.g., SHAP or LIME).
- Enhance the user interface for both CLI and web-based interactions.
- Set up CI/CD for automated testing and deployment.

Contact:
--------
For any issues, questions, or contributions, please open an issue or submit a pull request on GitHub.

Thank you for using the Housing Price Prediction Project!
