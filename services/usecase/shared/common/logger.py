"""
Centralized logging configuration for all modules.
Environment-aware logger that can be configured via ENV variables.
"""
import logging
import sys
import os
from pathlib import Path


def setup_logger(
    name: str = __name__, 
    log_file: str = None, 
    level: str = None,
    log_format: str = None
) -> logging.Logger:
    """
    Setup and configure logger with both file and console handlers.
    All parameters can be overridden by environment variables.
    
    Args:
        name: Logger name (typically __name__)
        log_file: Path to log file (default: from LOG_FILE env or 'app.log')
        level: Logging level (default: from LOG_LEVEL env or 'INFO')
        log_format: Log format string (default: from LOG_FORMAT env or standard format)
    
    Returns:
        Configured logger instance
    
    Environment Variables:
        LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        LOG_FILE: Path to log file
        LOG_FORMAT: Custom log format string
    """
    # Get configuration from environment or use defaults
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    
    if log_file is None:
        log_file = os.getenv("LOG_FILE", "app.log")
    
    if log_format is None:
        log_format = os.getenv(
            "LOG_FORMAT", 
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    
    # File handler
    try:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # If file logging fails, continue with console only
        console_handler.setLevel(logging.WARNING)
        logger.warning(f"Could not create file handler for {log_file}: {e}")
    
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger for a module.
    
    Args:
        name: Module name (use __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Default application logger (can be used directly)
app_logger = setup_logger("app")
