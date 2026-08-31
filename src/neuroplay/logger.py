"""
Centralized logging configuration for NeuroPlay-AI.
Provides a consistent logger (console + file output) across all modules.
"""

import logging
import sys

from neuroplay.config import settings


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance.
    Safe to call multiple times with the same name (won't duplicate handlers).
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # avoid duplicate handlers on re-import

    log_level = logging.DEBUG if settings.debug else logging.INFO
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(settings.logs_dir / "neuroplay.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
