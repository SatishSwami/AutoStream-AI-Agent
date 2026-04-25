"""
Centralized structured logger for AutoStream Agent.
Uses Python's logging module with coloured console output and optional file logging.
"""

import logging
import os
import sys
from datetime import datetime


LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.environ.get("LOG_FILE", None)  # Set to a path to enable file logging


class _ColorFormatter(logging.Formatter):
    """ANSI colour codes for terminal output."""
    GREY   = "\x1b[38;5;240m"
    CYAN   = "\x1b[36m"
    YELLOW = "\x1b[33m"
    RED    = "\x1b[31m"
    BOLD_RED = "\x1b[31;1m"
    RESET  = "\x1b[0m"

    FORMATS = {
        logging.DEBUG:    GREY    + "[%(asctime)s] [DEBUG]   %(name)s — %(message)s" + RESET,
        logging.INFO:     CYAN    + "[%(asctime)s] [INFO]    %(name)s — %(message)s" + RESET,
        logging.WARNING:  YELLOW  + "[%(asctime)s] [WARNING] %(name)s — %(message)s" + RESET,
        logging.ERROR:    RED     + "[%(asctime)s] [ERROR]   %(name)s — %(message)s" + RESET,
        logging.CRITICAL: BOLD_RED + "[%(asctime)s] [CRIT]   %(name)s — %(message)s" + RESET,
    }

    def format(self, record):
        fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(_ColorFormatter())
    logger.addHandler(ch)

    # Optional file handler
    if LOG_FILE:
        fh = logging.FileHandler(LOG_FILE)
        fh.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(fh)

    logger.propagate = False
    return logger
