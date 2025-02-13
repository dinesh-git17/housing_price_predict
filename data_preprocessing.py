from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from logger import logger  # Import logger


def load_data():
    logger.info("📊 Fetching California housing dataset...")
    data = fetch_california_housing()

    logger.info("🔀 Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    logger.info("✅ Data loading complete!")
    return X_train, X_test, y_train, y_test
