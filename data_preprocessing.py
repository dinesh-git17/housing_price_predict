# data_preprocessing.py

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def preprocess_data():
    # Load the dataset
    data = fetch_california_housing()

    # Convert to pandas DataFrame for easier manipulation
    import pandas as pd
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Split into features and target
    X = df.drop(columns=['target'])  # Features
    y = df['target']  # Target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
