"""Structured logging setup."""

import logging
import sys
from pathlib import Path


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Configure application-wide logging."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("strata")
    logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)-20s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(f"{log_dir}/strata_ml.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
