"""
File reading operations for Linux AI Agent.
"""

import os
from pathlib import Path
from typing import Optional, List
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utils import get_logger, log_operation


logger = get_logger("crud_read")


def read_file(filepath: str, encoding: str = 'utf-8') -> Optional[str]:
    """
    Read content from a file.
    
    Args:
        filepath: Path to the file to read
        encoding: File encoding
    
    Returns:
        File content as string, None if failed
    """
    try:
        file_path = Path(filepath)
        
        if not file_path.exists():
            logger.warning(f"File {filepath} does not exist")
            return None
        
        if not file_path.is_file():
            logger.warning(f"{filepath} is not a file")
            return None
        
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        
        log_operation(logger, "READ_FILE", {
            "filepath": str(file_path),
            "content_length": len(content),
            "encoding": encoding
        })
        
        return content
        
    except Exception as e:
        logger.error(f"Failed to read file {filepath}: {e}")
        return None


def read_binary_file(filepath: str) -> Optional[bytes]:
    """
    Read binary content from a file.
    
    Args:
        filepath: Path to the file to read
    
    Returns:
        File content as bytes, None if failed
    """
    try:
        file_path = Path(filepath)
        
        if not file_path.exists():
            logger.warning(f"File {filepath} does not exist")
            return None
        
        with open(file_path, 'rb') as f:
            content = f.read()
        
        log_operation(logger, "READ_BINARY_FILE", {
            "filepath": str(file_path),
            "content_length": len(content)
        })
        
        return content
        
    except Exception as e:
        logger.error(f"Failed to read binary file {filepath}: {e}")
        return None


def list_directory(dirpath: str, recursive: bool = False) -> List[str]:
    """
    List files and directories in a path.
    
    Args:
        dirpath: Path to the directory to list
        recursive: Whether to list recursively
    
    Returns:
        List of file/directory paths
    """
    try:
        dir_path = Path(dirpath)
        
        if not dir_path.exists():
            logger.warning(f"Directory {dirpath} does not exist")
            return []
        
        if not dir_path.is_dir():
            logger.warning(f"{dirpath} is not a directory")
            return []
        
        if recursive:
            items = [str(p) for p in dir_path.rglob('*')]
        else:
            items = [str(p) for p in dir_path.iterdir()]
        
        log_operation(logger, "LIST_DIRECTORY", {
            "dirpath": str(dir_path),
            "recursive": recursive,
            "item_count": len(items)
        })
        
        return items
        
    except Exception as e:
        logger.error(f"Failed to list directory {dirpath}: {e}")
        return []


def get_file_info(filepath: str) -> Optional[dict]:
    """
    Get file information including size, modification time, etc.
    
    Args:
        filepath: Path to the file
    
    Returns:
        Dictionary with file information, None if failed
    """
    try:
        file_path = Path(filepath)
        
        if not file_path.exists():
            logger.warning(f"File {filepath} does not exist")
            return None
        
        stat = file_path.stat()
        
        info = {
            "path": str(file_path),
            "size": stat.st_size,
            "modified_time": stat.st_mtime,
            "created_time": stat.st_ctime,
            "is_file": file_path.is_file(),
            "is_directory": file_path.is_dir(),
            "permissions": oct(stat.st_mode)[-3:]
        }
        
        log_operation(logger, "GET_FILE_INFO", {
            "filepath": str(file_path),
            "size": info["size"]
        })
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get file info for {filepath}: {e}")
        return None
