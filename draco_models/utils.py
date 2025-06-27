import logging
import pathlib
import sys

LOGGER_LEVEL: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def create_logger(
    name: str, log_file: str | pathlib.Path, level: str = "INFO"
) -> logging.Logger:
    """
    Create a logger with the given name and log file.

    Args:
        name (str): The name of the logger.
        log_file (str | pathlib.Path): Path to the log file.
        level (str): The logging level ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]. Defaults to "INFO".

    Returns:
        logging.Logger: The logger instance.
    """
    # Get the logging level
    level_log = LOGGER_LEVEL[level]
    # Create the logger and set it to the desired level
    logger = logging.getLogger(name)
    logger.setLevel(level_log)
    # Output file and log format
    FORMAT = logging.Formatter(
        "%(asctime)s - %(filename)s->%(funcName)s():%(lineno)s - [%(levelname)s] - %(message)s"
    )
    file_handler = logging.FileHandler(log_file, mode="w", encoding=None, delay=False)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMAT)
    file_handler.setFormatter(FORMAT)
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)  # Add console handler
    # Return the logger
    return logger
