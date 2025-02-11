# data_preprocessing.py
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data():
    # Fetch the California housing dataset
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target  # Add target column

    # Split the data into features (X) and target (y)
    X = df.drop(columns=['target'])  # Drop target column
    y = df['target']  # Extract target column
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test  # Correctly return these four variables
