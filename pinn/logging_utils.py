from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(logger_name: str, log_file: Path, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.propagate = False

    level_value = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level_value)

    if logger.handlers:
        for handler in logger.handlers:
            handler.setLevel(level_value)
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level_value)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level_value)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
