"""
Logger configuration module
"""

import sys
import time
import logging

from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "logs"
FORMATTER = logging.Formatter("[%(asctime)s] [%(thread)d] %(name)s %(levelname)s: %(message)s")


def configure_logger(name=None):
    """
    Prepare logger
    :return: logger
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_time = time.strftime("%d%m_%I%M%S")

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(FORMATTER)

    file_handler = logging.FileHandler(f"{LOG_PATH}\\{name.replace('.', '-')}_{log_time}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(FORMATTER)

    # noinspection PyArgumentList
    logging.basicConfig(
        level=logging.INFO,
        datefmt='%d/%m/%Y %I:%M:%S %p',
        handlers=[stream_handler, file_handler]
    )
    return logger