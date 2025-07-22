"""
File creation operations for Linux AI Agent.
"""

import os
from pathlib import Path
from typing import Optional
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utils import get_logger, log_operation


logger = get_logger("crud_create")


def create_file(filepath: str, content: str = "", overwrite: bool = False) -> bool:
    """
    Create a new file with optional content.
    
    Args:
        filepath: Path to the file to create
        content: Content to write to the file
        overwrite: Whether to overwrite if file exists
    
    Returns:
        True if successful, False otherwise
    """
    try:
        file_path = Path(filepath)
        
        # Check if file exists and overwrite is False
        if file_path.exists() and not overwrite:
            logger.warning(f"File {filepath} already exists and overwrite=False")
            return False
        
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        log_operation(logger, "CREATE_FILE", {
            "filepath": str(file_path),
            "content_length": len(content),
            "overwrite": overwrite
        })
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create file {filepath}: {e}")
        return False


def create_directory(dirpath: str) -> bool:
    """
    Create a new directory.
    
    Args:
        dirpath: Path to the directory to create
    
    Returns:
        True if successful, False otherwise
    """
    try:
        dir_path = Path(dirpath)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        log_operation(logger, "CREATE_DIRECTORY", {
            "dirpath": str(dir_path)
        })
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create directory {dirpath}: {e}")
        return False
