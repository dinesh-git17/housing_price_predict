import joblib
from logger import logger

def load_model(model_filename="random_forest_model.pkl"):
    try:
        model = joblib.load(model_filename)
        logger.info(f"✅ Model loaded successfully from {model_filename}")
        return model
    except FileNotFoundError:
        logger.error(f"❌ Model file '{model_filename}' not found. Train the model first.")
        return None

# Test loading
if __name__ == "__main__":
    model = load_model()
    if model:
        logger.info("🎯 Model is ready for predictions!")
