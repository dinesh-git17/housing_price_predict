import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.datasets import fetch_california_housing


@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_random_forest_model.pkl")
        return model
    except Exception as e:
        st.error("Model not found. Please train the model first.")
        return None


def main():
    st.title("Housing Price Prediction Dashboard")
    st.write(
        "Enter the following details about the housing block to predict the median house price."
    )

    # Get dataset details
    data = fetch_california_housing()
    feature_names = data.feature_names

    # Create a mapping of feature names to descriptive labels
    feature_descriptions = {
        "MedInc": "Median Income (in tens of thousands)",
        "HouseAge": "Median House Age (years)",
        "AveRooms": "Average Number of Rooms per Household",
        "AveBedrms": "Average Number of Bedrooms per Household",
        "Population": "Population of the Block Group",
        "AveOccup": "Average Household Size (Occupancy)",
        "Latitude": "Latitude Coordinate",
        "Longitude": "Longitude Coordinate",
    }

    inputs = {}
    for feature in feature_names:
        default_value = float(np.mean(data.data[:, data.feature_names.index(feature)]))
        label = feature_descriptions.get(feature, feature)
        inputs[feature] = st.number_input(f"{label}:", value=default_value)

    if st.button("Predict"):
        model = load_model()
        if model:
            input_df = pd.DataFrame([list(inputs.values())], columns=feature_names)
            prediction = model.predict(input_df.to_numpy())
            st.success(f"Predicted House Price: ${prediction[0]*100000:.2f}")

    st.markdown("---")
    st.write("When you're done, you can exit the dashboard using the button below.")
    if st.button("Exit Dashboard"):
        st.write("Exiting dashboard...")
        os._exit(0)


if __name__ == "__main__":
    main()
