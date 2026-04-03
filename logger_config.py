# logger_config.py — Centralized Logging Configuration for CropAI
# Provides structured logging with file rotation and colored console output.

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from config import LOGS_DIR

os.makedirs(LOGS_DIR, exist_ok=True)

# ─── Formatter ────────────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s │ %(levelname)-8s │ %(name)-20s │ %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ─── Color codes for console output ──────────────────────────────────────────

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds ANSI color codes to log level names."""

    COLORS = {
        "DEBUG":    "\033[36m",    # Cyan
        "INFO":     "\033[32m",    # Green
        "WARNING":  "\033[33m",    # Yellow
        "ERROR":    "\033[31m",    # Red
        "CRITICAL": "\033[1;31m",  # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


# ─── Logger Factory ──────────────────────────────────────────────────────────

def get_logger(name: str, log_file: str = "cropai.log",
               level: int = logging.INFO) -> logging.Logger:
    """
    Create a logger with both console and rotating file handlers.

    Args:
        name:     Logger name (typically module name)
        log_file: Log file name inside LOGS_DIR
        level:    Logging level

    Returns:
        Configured logger instance

    Usage:
        from logger_config import get_logger
        logger = get_logger(__name__)
        logger.info("Model loaded successfully")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # ── Console handler (colored) ──────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(console_handler)

    # ── File handler (rotating, 5 MB × 3 backups) ─────────────────────────
    log_path = os.path.join(LOGS_DIR, log_file)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(file_handler)

    return logger


# ─── Pre-configured loggers ──────────────────────────────────────────────────

def get_api_logger():
    """Logger for the FastAPI backend."""
    return get_logger("CropAI.API", "api.log")

def get_training_logger():
    """Logger for model training loops."""
    return get_logger("CropAI.Training", "training.log")

def get_inference_logger():
    """Logger for prediction / inference."""
    return get_logger("CropAI.Inference", "inference.log")

def get_data_logger():
    """Logger for data processing / generation."""
    return get_logger("CropAI.Data", "data.log")
