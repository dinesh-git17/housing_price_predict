import folium
import pandas as pd
from sklearn.datasets import fetch_california_housing


def create_map():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["Target"] = data.target

    # Create a map centered on California
    m = folium.Map(location=[37.0, -120.0], zoom_start=6)

    # Plot the first 100 data points as an example
    for idx, row in df.head(100).iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=5,
            popup=f"Value: ${row['Target']*100000:.2f}",
            color="blue",
            fill=True,
            fill_color="blue",
        ).add_to(m)

    m.save("california_housing_map.html")
    print("Map saved as california_housing_map.html")


if __name__ == "__main__":
    create_map()
