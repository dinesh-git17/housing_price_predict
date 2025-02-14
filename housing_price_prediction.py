import sys
import os
from data_preprocessing import load_data
from model_training import train_and_evaluate_model
from model_tuning import tune_model
from model_evaluation import evaluate_model
from model_loader import load_model
import joblib
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def quiet_input(prompt):
    """
    Temporarily redirects sys.stderr to suppress verbose outputs (like IMKClient messages on macOS)
    when prompting the user.
    """
    import sys
    import os

    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        value = input(prompt)
    finally:
        sys.stderr = original_stderr
    return value


def run_pipeline(console):
    console.print(
        Panel.fit(
            Text("📥 Loading dataset...", style="bold blue"),
            title="Data Loading",
            border_style="blue",
        )
    )
    X_train, X_test, y_train, y_test = load_data()

    console.print(
        Panel.fit(
            Text("🔍 Checking for pre-trained model...", style="bold yellow"),
            title="Model Loading",
            border_style="yellow",
        )
    )
    model = load_model("best_random_forest_model.pkl")
    if model:
        choice = (
            quiet_input(
                "A saved model is available. Do you want to use the saved model? (y/n): "
            )
            .strip()
            .lower()
        )
        if choice != "y":
            console.print(
                Panel.fit(
                    Text("Retraining new model as per user request.", style="bold red"),
                    title="Retrain Model",
                    border_style="red",
                )
            )
            model = None

    if model:
        console.print(
            Panel.fit(
                Text(
                    "✅ Pre-trained model found! Evaluating on test set...",
                    style="bold green",
                ),
                title="Model Found",
                border_style="green",
            )
        )
        evaluate_model(model, X_test, y_test)
    else:
        console.print(
            Panel.fit(
                Text(
                    "⚠️ No pre-trained model found or retraining requested. Training a new model...",
                    style="bold red",
                ),
                title="Training Required",
                border_style="red",
            )
        )
        console.print(
            Panel.fit(
                Text("🚀 Training initial model...", style="bold green"),
                title="Model Training",
                border_style="green",
            )
        )
        initial_model, initial_rmse, initial_r2 = train_and_evaluate_model(
            X_train, X_test, y_train, y_test
        )
        console.print(
            Panel.fit(
                Text(
                    f"🏆 Initial Model Performance: RMSE={initial_rmse:.4f}, R²={initial_r2:.4f}",
                    style="bold cyan",
                ),
                title="Initial Model Results",
                border_style="cyan",
            )
        )
        console.print(
            Panel.fit(
                Text("🔍 Performing hyperparameter tuning...", style="bold magenta"),
                title="Hyperparameter Tuning",
                border_style="magenta",
            )
        )
        best_model, best_rmse, best_r2 = tune_model(X_train, y_train)
        console.print(
            Panel.fit(
                Text(
                    f"🏆 Best Model Performance: RMSE={best_rmse:.4f}, R²={best_r2:.4f}",
                    style="bold yellow",
                ),
                title="Best Model Results",
                border_style="yellow",
            )
        )
        best_model_filename = "best_random_forest_model.pkl"
        joblib.dump(best_model, best_model_filename)
        console.print(
            Panel.fit(
                Text(
                    f"💾 Best model saved as {best_model_filename}", style="bold green"
                ),
                title="Model Saved",
                border_style="green",
            )
        )
        console.print(
            Panel.fit(
                Text(
                    "✅ Model tuning complete! Evaluating final model...",
                    style="bold blue",
                ),
                title="Final Evaluation",
                border_style="blue",
            )
        )
        evaluate_model(best_model, X_test, y_test)

    user_choice = (
        quiet_input("Do you want to make predictions using the trained model? (y/n): ")
        .strip()
        .lower()
    )
    if user_choice == "y":
        try:
            from predict import interactive_mode

            interactive_mode()
        except ImportError:
            console.print(
                Panel.fit(
                    "[bold red]Error: Could not import interactive prediction mode from predict.py[/bold red]",
                    title="Error",
                    border_style="red",
                )
            )
    else:
        console.print(
            Panel.fit(
                "[bold blue]Skipping predictions.[/bold blue]",
                title="Exit",
                border_style="blue",
            )
        )


def main_menu():
    console = Console()
    while True:
        console.print(
            Panel.fit(
                "[bold magenta]Welcome to the Housing Price Prediction Project[/bold magenta]",
                title="Main Menu",
                border_style="magenta",
            )
        )
        console.print("Choose an option:")
        console.print("1. Run Pipeline (Train, Evaluate, Predict)")
        console.print("2. Visualize Data (Histograms & Scatter Plots)")
        console.print("3. Generate Geospatial Map")
        console.print("4. Launch Interactive Dashboard (Streamlit)")
        console.print("5. Exit")
        choice = quiet_input("Enter your choice (1/2/3/4/5): ").strip()
        if choice == "1":
            run_pipeline(console)
        elif choice == "2":
            try:
                import visualize

                visualize.run_all_visualizations()
            except ImportError as e:
                console.print(
                    Panel.fit(
                        f"[bold red]Error: {e}[/bold red]",
                        title="Error",
                        border_style="red",
                    )
                )
        elif choice == "3":
            try:
                import geoviz

                geoviz.create_map()
            except ImportError as e:
                console.print(
                    Panel.fit(
                        f"[bold red]Error: {e}[/bold red]",
                        title="Error",
                        border_style="red",
                    )
                )
        elif choice == "4":
            console.print(
                Panel.fit(
                    "[bold green]Launching Streamlit dashboard...[/bold green]",
                    title="Dashboard",
                    border_style="green",
                )
            )
            os.system("streamlit run app.py")
        elif choice == "5":
            more = (
                quiet_input(
                    "Do you want to return to the main menu before quitting? (y/n): "
                )
                .strip()
                .lower()
            )
            if more == "y":
                continue
            else:
                console.print(
                    Panel.fit(
                        "[bold blue]Goodbye![/bold blue]",
                        title="Exit",
                        border_style="blue",
                    )
                )
                sys.exit(0)
        else:
            console.print(
                Panel.fit(
                    "[bold red]Invalid choice. Try again.[/bold red]",
                    title="Error",
                    border_style="red",
                )
            )


if __name__ == "__main__":
    main_menu()
