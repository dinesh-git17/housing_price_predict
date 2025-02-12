from data_preprocessing import load_data
from model_training import train_and_evaluate_model
from model_tuning import tune_model
from model_evaluation import evaluate_model
import joblib
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def main():
    console = Console()  # Create a Rich console instance

    try:
        # Step 1: Load data
        console.print(Panel.fit(
            Text("📥 Loading dataset...", style="bold blue"),
            title="Data Loading",
            border_style="blue"
        ))
        X_train, X_test, y_train, y_test = load_data()

        # Step 2: Train and evaluate initial model
        console.print(Panel.fit(
            Text("🚀 Training initial model...", style="bold green"),
            title="Model Training",
            border_style="green"
        ))
        initial_model, initial_rmse, initial_r2 = train_and_evaluate_model(X_train, X_test, y_train, y_test)
        console.print(Panel.fit(
            Text(f"🏆 Initial Model Performance: RMSE={initial_rmse:.4f}, R²={initial_r2:.4f}", style="bold cyan"),
            title="Initial Model Results",
            border_style="cyan"
        ))

        # Step 3: Perform hyperparameter tuning
        console.print(Panel.fit(
            Text("🔍 Performing hyperparameter tuning...", style="bold magenta"),
            title="Hyperparameter Tuning",
            border_style="magenta"
        ))
        best_model, best_rmse, best_r2 = tune_model(X_train, y_train)
        console.print(Panel.fit(
            Text(f"🏆 Best Model Performance: RMSE={best_rmse:.4f}, R²={best_r2:.4f}", style="bold yellow"),
            title="Best Model Results",
            border_style="yellow"
        ))

        # Step 4: Save the best model
        best_model_filename = "best_random_forest_model.pkl"
        joblib.dump(best_model, best_model_filename)
        console.print(Panel.fit(
            Text(f"💾 Best model saved as {best_model_filename}", style="bold green"),
            title="Model Saved",
            border_style="green"
        ))

        # Step 5: Evaluate the best model on the test set
        console.print(Panel.fit(
            Text("✅ Model tuning complete! Evaluating final model...", style="bold blue"),
            title="Final Evaluation",
            border_style="blue"
        ))
        evaluate_model(best_model, X_test, y_test)

    except Exception as e:
        console.print(Panel.fit(
            Text(f"❌ An error occurred: {e}", style="bold red"),
            title="Error",
            border_style="red"
        ))


if __name__ == "__main__":
    main()
