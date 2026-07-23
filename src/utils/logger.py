"""
Centralized logging configuration for the Recommendation System.

Uses loguru for structured, colored, rotating log output.
All modules should import `get_logger` from this module rather
than configuring their own loggers.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Model training started")
    logger.error("Failed to load data: {error}", error=str(e))
"""

import sys
from pathlib import Path
from typing import Optional

# pyrefly: ignore [missing-import]
from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    compression: str = "zip",
) -> None:
    """
    Configure loguru logger with console and optional file handlers.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to the log file. If None, only console output is used.
        rotation: When to rotate the log file (size or time, e.g. "10 MB", "1 day").
        retention: How long to keep old log files (e.g. "7 days").
        compression: Compression format for rotated files ("zip", "gz", etc.).
    """
    # Remove any existing default handlers
    logger.remove()

    # ── Console handler ────────────────────────────────────────────────────────
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stdout,
        format=console_format,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # ── File handler (optional) ────────────────────────────────────────────────
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} - "
            "{message}"
        )
        logger.add(
            str(log_path),
            format=file_format,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression=compression,
            backtrace=True,
            diagnose=True,
            enqueue=True,   # thread-safe writing
        )


def get_logger(name: str) -> "logger.__class__":
    """
    Return a named loguru logger instance bound to a specific module.

    This binds the module name so log records include the source context
    even though loguru uses a single global logger internally.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A loguru logger instance with the module name bound.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Initializing model")
    """
    return logger.bind(name=name)


# ── Default initialisation ─────────────────────────────────────────────────────
# Called once at import time with safe defaults.
# The trainer and API main.py call setup_logging() again with config values.
setup_logging(log_level="INFO")
