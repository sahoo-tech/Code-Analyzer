# Logging configuration

import logging
import sys
from typing import Optional
from pathlib import Path
from logging.handlers import RotatingFileHandler


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    root = logging.getLogger("analyzer")
    root.setLevel(getattr(logging, level.upper()))
    
    root.handlers.clear()
    
    fmt = format_string or "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ColoredFormatter(fmt))
    root.addHandler(console)
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=3
        )
        file_handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    if not name.startswith("analyzer"):
        name = f"analyzer.{name}"
    return logging.getLogger(name)


configure_logging()
