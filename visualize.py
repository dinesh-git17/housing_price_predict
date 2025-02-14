import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
import pandas as pd


def plot_histograms():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["Target"] = data.target

    plt.figure(figsize=(8, 6))
    sns.histplot(df["MedInc"], kde=True, color="blue")
    plt.title("Distribution of Median Income")
    plt.xlabel("Median Income")
    plt.ylabel("Frequency")
    plt.show()


def plot_scatter():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["Target"] = data.target

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df["HouseAge"], y=df["Target"])
    plt.title("House Age vs. Median House Value")
    plt.xlabel("House Age")
    plt.ylabel("Median House Value")
    plt.show()


def run_all_visualizations():
    plot_histograms()
    plot_scatter()


if __name__ == "__main__":
    run_all_visualizations()
