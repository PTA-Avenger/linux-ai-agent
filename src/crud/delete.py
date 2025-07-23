"""
File deletion operations for Linux AI Agent.
"""

import os
import shutil
from pathlib import Path
from typing import List
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utils import get_logger, log_operation


logger = get_logger("crud_delete")


def delete_file(filepath: str, force: bool = False) -> bool:
    """
    Delete a file.
    
    Args:
        filepath: Path to the file to delete
        force: Whether to force deletion (ignore permission errors)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        file_path = Path(filepath)
        
        if not file_path.exists():
            logger.warning(f"File {filepath} does not exist")
            return False
        
        if not file_path.is_file():
            logger.warning(f"{filepath} is not a file")
            return False
        
        # Get file info before deletion
        size = file_path.stat().st_size
        
        if force:
            # Change permissions to allow deletion if needed
            file_path.chmod(0o777)
        
        file_path.unlink()
        
        log_operation(logger, "DELETE_FILE", {
            "filepath": str(file_path),
            "size": size,
            "force": force
        })
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete file {filepath}: {e}")
        return False


def delete_directory(dirpath: str, recursive: bool = False, force: bool = False) -> bool:
    """
    Delete a directory.
    
    Args:
        dirpath: Path to the directory to delete
        recursive: Whether to delete recursively (with contents)
        force: Whether to force deletion
    
    Returns:
        True if successful, False otherwise
    """
    try:
        dir_path = Path(dirpath)
        
        if not dir_path.exists():
            logger.warning(f"Directory {dirpath} does not exist")
            return False
        
        if not dir_path.is_dir():
            logger.warning(f"{dirpath} is not a directory")
            return False
        
        # Count items before deletion
        items = list(dir_path.iterdir())
        item_count = len(items)
        
        if not recursive and item_count > 0:
            logger.warning(f"Directory {dirpath} is not empty and recursive=False")
            return False
        
        if recursive:
            if force:
                # Change permissions recursively if needed
                for root, dirs, files in os.walk(dirpath):
                    for d in dirs:
                        os.chmod(os.path.join(root, d), 0o777)
                    for f in files:
                        os.chmod(os.path.join(root, f), 0o777)
            
            shutil.rmtree(dir_path)
        else:
            dir_path.rmdir()
        
        log_operation(logger, "DELETE_DIRECTORY", {
            "dirpath": str(dir_path),
            "item_count": item_count,
            "recursive": recursive,
            "force": force
        })
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete directory {dirpath}: {e}")
        return False


def delete_files_by_pattern(directory: str, pattern: str) -> List[str]:
    """
    Delete files matching a pattern.
    
    Args:
        directory: Directory to search in
        pattern: File pattern (glob style)
    
    Returns:
        List of deleted file paths
    """
    deleted_files = []
    
    try:
        dir_path = Path(directory)
        
        if not dir_path.exists() or not dir_path.is_dir():
            logger.warning(f"Directory {directory} does not exist or is not a directory")
            return deleted_files
        
        # Find matching files
        matching_files = list(dir_path.glob(pattern))
        
        for file_path in matching_files:
            if file_path.is_file():
                try:
                    file_path.unlink()
                    deleted_files.append(str(file_path))
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")
        
        log_operation(logger, "DELETE_BY_PATTERN", {
            "directory": str(dir_path),
            "pattern": pattern,
            "deleted_count": len(deleted_files)
        })
        
    except Exception as e:
        logger.error(f"Failed to delete files by pattern in {directory}: {e}")
    
    return deleted_files


def secure_delete(filepath: str, passes: int = 3) -> bool:
    """
    Securely delete a file by overwriting it multiple times.
    
    Args:
        filepath: Path to the file to securely delete
        passes: Number of overwrite passes
    
    Returns:
        True if successful, False otherwise
    """
    try:
        file_path = Path(filepath)
        
        if not file_path.exists():
            logger.warning(f"File {filepath} does not exist")
            return False
        
        if not file_path.is_file():
            logger.warning(f"{filepath} is not a file")
            return False
        
        size = file_path.stat().st_size
        
        # Overwrite file multiple times
        with open(file_path, 'r+b') as f:
            for i in range(passes):
                f.seek(0)
                # Write random data
                import random
                random_data = bytes([random.randint(0, 255) for _ in range(size)])
                f.write(random_data)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
        
        # Finally delete the file
        file_path.unlink()
        
        log_operation(logger, "SECURE_DELETE", {
            "filepath": str(file_path),
            "size": size,
            "passes": passes
        })
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to securely delete file {filepath}: {e}")
        return False
