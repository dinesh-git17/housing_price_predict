import logging
import colorlog

# Create a logger
logger = logging.getLogger(__name__)

# Prevent duplicate handlers
if not logger.hasHandlers():
    handler = logging.StreamHandler()

    log_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
        log_colors={
            "DEBUG": "white",
            "INFO": "light_purple",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red"
        }
    )

    handler.setFormatter(log_formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
