"""
Utility functions for logging.
"""

import logging
from pathlib import Path
from typing import Dict

import pandas as pd


def save_error_scores(csv_path: Path, value: Dict, error_type: str) -> None:
    """
    Save error scores to a CSV file.

    Args:
        csv_path: Path: The path to the CSV file.
        value: Dict: The value to save.
        error_type: str: The type of error.
    """
    score = pd.DataFrame.from_dict(value, orient="index")
    score.reset_index(inplace=True)
    column_means = score.iloc[:, 1:].mean()

    # Append the means to the dataframe as the last row
    # Set the 'index' column of the new row to a specific identifier
    means_row = pd.DataFrame([["mean"] + column_means.tolist()], columns=score.columns)
    score = pd.concat([score, means_row], ignore_index=True)
    # score.loc[len(score.index)] = ['avg', score.iloc[:, 1:].mean().tolist()]#sum(value.values()) / len(value)]
    # Write DataFrame to CSV
    score.to_csv(f"{csv_path}/{error_type}.csv", index=False)


def get_logger(log_dir: Path | None = None) -> logging.Logger:
    """
    Create a logger that logs to both a file and the console.
    :param log_dir: Directory where the log file will be saved.
    :return: Configured logger instance.
    """
    # Create a logger
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)

    # Create a console handler to log to the console
    console_handler = logging.StreamHandler()

    # Set the log format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)

    # Create a file handler to log to a file
    if log_dir is not None:
        log_file = log_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger_initialized() -> logging.Logger:
    """
    Get a logger that is already initialized with a specific configuration.
    This function is useful for ensuring that the logger is set up correctly
    before it is used in other parts of the code.

    Returns:
        logging.Logger: The initialized logger.
    """
    logger = logging.getLogger("training_logger")
    # check if the logger has at least 2 handlers (file and console)
    # if not, it means it has not been initialized yet
    if not logger.hasHandlers():
        logger = get_logger(log_dir=None)
        logger.info("Logger initialized without writing to file/log_dir.")
    return logger
