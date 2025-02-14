import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from model_loader import load_model
from logger import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def preprocess_input(user_input):
    """
    Ensures the input data is formatted correctly for the model.
    Assumes input data matches the structure of the California housing dataset.
    """
    data = fetch_california_housing()  # Get feature names
    feature_names = data.feature_names

    if isinstance(user_input, list):
        user_input = np.array(user_input).reshape(
            1, -1
        )  # Reshape for a single prediction
        df = pd.DataFrame(user_input, columns=feature_names)
    elif isinstance(user_input, pd.DataFrame):
        df = user_input
    else:
        raise ValueError("Input must be a list of values or a DataFrame.")
    return df


def make_prediction(input_data):
    """Loads the model and makes a prediction on the given input data."""
    model = load_model("best_random_forest_model.pkl")

    if model is None:
        console.print(
            Panel.fit(
                "[bold red]❌ No trained model found. Train the model first.[/bold red]",
                title="Error",
            )
        )
        return None

    try:
        processed_data = preprocess_input(input_data)
        # Convert DataFrame to numpy array to avoid feature names conflict
        if isinstance(processed_data, pd.DataFrame):
            processed_data = processed_data.to_numpy()
        predictions = model.predict(processed_data)
        return predictions
    except Exception as e:
        logger.error(f"❌ Error making prediction: {e}")
        return None


def interactive_mode():
    instructions = (
        "Prediction Instructions:\n\n"
        "For a single prediction, please provide 8 numeric feature values in the following order:\n"
        "  1. MedInc     - Median income in the block group\n"
        "  2. HouseAge   - Median house age in the block group\n"
        "  3. AveRooms   - Average number of rooms per household\n"
        "  4. AveBedrms  - Average number of bedrooms per household\n"
        "  5. Population - Population of the block group\n"
        "  6. AveOccup   - Average occupancy (household size)\n"
        "  7. Latitude   - Latitude coordinate\n"
        "  8. Longitude  - Longitude coordinate\n\n"
        "Alternatively, you can provide a CSV file containing these columns:\n"
        "MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude\n"
    )
    console.print(
        Panel.fit(instructions, title="Prediction Instructions", border_style="blue")
    )

    choice = input(
        "Enter '1' for a single prediction or '2' for a CSV file for batch predictions: "
    ).strip()
    if choice == "1":
        values = input("Enter 8 feature values separated by commas: ").strip()
        try:
            values = [float(v.strip()) for v in values.split(",")]
        except Exception as e:
            console.print(
                Panel.fit(
                    f"[bold red]Error parsing input: {e}[/bold red]", title="Error"
                )
            )
            return
        if len(values) != 8:
            console.print(
                Panel.fit(
                    "[bold red]Error: Exactly 8 feature values are required.[/bold red]",
                    title="Error",
                )
            )
            return
        predictions = make_prediction(values)
        if predictions is not None:
            console.print(
                Panel.fit(
                    f"🏡 Predicted House Price: [bold green]${predictions[0]*100000:.2f}[/bold green]",
                    title="Prediction",
                    border_style="green",
                )
            )
    elif choice == "2":
        csv_path = input("Enter the path to your CSV file: ").strip()
        try:
            df = pd.read_csv(csv_path)
            predictions = make_prediction(df)
            if predictions is not None:
                table = Table(title="Predictions")
                table.add_column("Index", justify="right", style="cyan", no_wrap=True)
                table.add_column("Predicted Price", style="green")
                for i, pred in enumerate(predictions):
                    table.add_row(str(i), f"${pred*100000:.2f}")
                console.print(table)
        except Exception as e:
            console.print(
                Panel.fit(
                    f"[bold red]Error reading CSV file: {e}[/bold red]",
                    title="Error",
                    border_style="red",
                )
            )
    else:
        console.print(
            Panel.fit("[bold red]Invalid choice. Exiting.[/bold red]", title="Error")
        )


def cli():
    if len(sys.argv) == 1:
        # No arguments provided, enter interactive mode
        interactive_mode()
    else:
        # Use argparse if command-line arguments are provided
        parser = argparse.ArgumentParser(
            description=(
                "Predict housing prices using a trained RandomForestRegressor model.\n\n"
                "For a single prediction, provide 8 numeric feature values in the following order:\n"
                "  1. MedInc     - Median income in the block group\n"
                "  2. HouseAge   - Median house age in the block group\n"
                "  3. AveRooms   - Average number of rooms per household\n"
                "  4. AveBedrms  - Average number of bedrooms per household\n"
                "  5. Population - Population of the block group\n"
                "  6. AveOccup   - Average occupancy (household size)\n"
                "  7. Latitude   - Latitude coordinate\n"
                "  8. Longitude  - Longitude coordinate\n\n"
                "Alternatively, provide a CSV file with these columns in the header."
            )
        )
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "-s",
            "--single",
            nargs="+",
            type=float,
            help="Space-separated list of 8 feature values for a single prediction.",
        )
        group.add_argument(
            "-c",
            "--csv",
            type=str,
            help="Path to a CSV file with columns: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude.",
        )
        args = parser.parse_args()

        if args.single:
            if len(args.single) != 8:
                console.print(
                    Panel.fit(
                        "[bold red]❌ Error: Exactly 8 feature values are required for a single prediction.[/bold red]",
                        title="Error",
                    )
                )
                return
            input_data = args.single
            predictions = make_prediction(input_data)
            if predictions is not None:
                console.print(
                    Panel.fit(
                        f"🏡 Predicted House Price: [bold green]${predictions[0]*100000:.2f}[/bold green]",
                        title="Prediction",
                        border_style="green",
                    )
                )
        elif args.csv:
            try:
                df = pd.read_csv(args.csv)
                predictions = make_prediction(df)
                if predictions is not None:
                    table = Table(title="Predictions")
                    table.add_column(
                        "Index", justify="right", style="cyan", no_wrap=True
                    )
                    table.add_column("Predicted Price", style="green")
                    for i, pred in enumerate(predictions):
                        table.add_row(str(i), f"${pred*100000:.2f}")
                    console.print(table)
            except Exception as e:
                console.print(
                    Panel.fit(
                        f"[bold red]❌ Error reading CSV file: {e}[/bold red]",
                        title="Error",
                        border_style="red",
                    )
                )


if __name__ == "__main__":
    cli()
