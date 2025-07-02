"""
gnn_trading.utils.logging
========================
Centralized logging configuration
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Setup centralized logging configuration"""
    
    if format_string is None:
        format_string = "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Setup root logger
    root_logger = logging.getLogger("gnn_trading")
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get logger with consistent naming"""
    return logging.getLogger(f"gnn_trading.{name}")
