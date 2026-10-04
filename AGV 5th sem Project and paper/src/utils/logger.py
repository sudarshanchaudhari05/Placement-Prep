"""
Research Experiment Logger
==========================
Sets up dual console and file logging with experiment metadata tracking.
"""

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "AGVPathTracking",
    log_dir: Optional[Path] = None,
    log_filename: str = "experiment.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure experiment logger.
    
    Args:
        name: Logger name.
        log_dir: Directory where log file will be saved.
        log_filename: Name of log file.
        level: Logging verbosity level.
        
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / log_filename, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
