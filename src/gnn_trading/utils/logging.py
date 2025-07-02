"""
gnn_trading.utils.logging
=========================

Enhanced logging configuration for the GNN Trading System
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup enhanced logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    
    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Default format
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Set library log levels to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger with specified name"""
    return logging.getLogger(name)


class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self, name: str = "performance"):
        self.logger = logging.getLogger(name)
        
    def log_timing(self, operation: str, duration: float, **kwargs):
        """Log timing information"""
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"TIMING: {operation} took {duration:.3f}s {extra_info}")
        
    def log_memory(self, operation: str, memory_mb: float, **kwargs):
        """Log memory usage"""
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"MEMORY: {operation} used {memory_mb:.1f}MB {extra_info}")
        
    def log_throughput(self, operation: str, items: int, duration: float, **kwargs):
        """Log throughput metrics"""
        throughput = items / duration if duration > 0 else 0
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"THROUGHPUT: {operation} processed {items} items in {duration:.3f}s "
                        f"({throughput:.1f} items/s) {extra_info}")


class TradingLogger:
    """Logger for trading-specific events"""
    
    def __init__(self, name: str = "trading"):
        self.logger = logging.getLogger(name)
        
    def log_trade(self, symbol: str, action: str, quantity: int, price: float, **kwargs):
        """Log trade execution"""
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"TRADE: {action} {quantity} {symbol} @ {price:.2f} {extra_info}")
        
    def log_signal(self, symbol: str, signal: str, strength: float, **kwargs):
        """Log trading signal"""
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"SIGNAL: {symbol} {signal} strength={strength:.3f} {extra_info}")
        
    def log_portfolio(self, total_value: float, cash: float, positions: dict, **kwargs):
        """Log portfolio state"""
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"PORTFOLIO: value={total_value:.2f} cash={cash:.2f} "
                        f"positions={len(positions)} {extra_info}")


class ModelLogger:
    """Logger for model training and inference"""
    
    def __init__(self, name: str = "model"):
        self.logger = logging.getLogger(name)
        
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, **kwargs):
        """Log training epoch"""
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"EPOCH {epoch}: train_loss={train_loss:.6f} "
                        f"val_loss={val_loss:.6f} {extra_info}")
        
    def log_prediction(self, model_id: str, prediction: float, confidence: float, **kwargs):
        """Log model prediction"""
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"PREDICTION: {model_id} pred={prediction:.6f} "
                        f"confidence={confidence:.3f} {extra_info}")
        
    def log_model_update(self, model_id: str, version: str, performance: float, **kwargs):
        """Log model update"""
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"MODEL_UPDATE: {model_id} v{version} "
                        f"performance={performance:.3f} {extra_info}")


# Create default loggers
performance_logger = PerformanceLogger()
trading_logger = TradingLogger()
model_logger = ModelLogger()
