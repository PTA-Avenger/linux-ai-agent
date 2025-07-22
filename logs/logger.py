"""
Logging utilities for Linux AI Agent.
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def get_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Create and configure a logger.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Avoid adding multiple handlers
    if not logger.handlers:
        # Create file handler
        log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


def log_operation(logger: logging.Logger, operation: str, details: dict):
    """
    Log an operation with structured details.
    
    Args:
        logger: Logger instance
        operation: Operation name
        details: Operation details dictionary
    """
    logger.info(f"Operation: {operation} | Details: {details}")
