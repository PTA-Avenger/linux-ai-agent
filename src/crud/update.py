"""
File update operations for Linux AI Agent.
"""

import os
from pathlib import Path
from typing import Optional
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utils import get_logger, log_operation


logger = get_logger("crud_update")


def update_file(filepath: str, content: str, mode: str = 'w', encoding: str = 'utf-8') -> bool:
    """
    Update a file with new content.
    
    Args:
        filepath: Path to the file to update
        content: New content to write
        mode: Write mode ('w' for overwrite, 'a' for append)
        encoding: File encoding
    
    Returns:
        True if successful, False otherwise
    """
    try:
        file_path = Path(filepath)
        
        # Check if file exists
        if not file_path.exists():
            logger.warning(f"File {filepath} does not exist")
            return False
        
        # Backup original content for logging
        original_size = file_path.stat().st_size
        
        with open(file_path, mode, encoding=encoding) as f:
            f.write(content)
        
        log_operation(logger, "UPDATE_FILE", {
            "filepath": str(file_path),
            "original_size": original_size,
            "new_content_length": len(content),
            "mode": mode,
            "encoding": encoding
        })
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to update file {filepath}: {e}")
        return False


def append_to_file(filepath: str, content: str, encoding: str = 'utf-8') -> bool:
    """
    Append content to a file.
    
    Args:
        filepath: Path to the file to append to
        content: Content to append
        encoding: File encoding
    
    Returns:
        True if successful, False otherwise
    """
    return update_file(filepath, content, mode='a', encoding=encoding)


def replace_in_file(filepath: str, old_text: str, new_text: str, encoding: str = 'utf-8') -> bool:
    """
    Replace text in a file.
    
    Args:
        filepath: Path to the file
        old_text: Text to replace
        new_text: Replacement text
        encoding: File encoding
    
    Returns:
        True if successful, False otherwise
    """
    try:
        file_path = Path(filepath)
        
        if not file_path.exists():
            logger.warning(f"File {filepath} does not exist")
            return False
        
        # Read current content
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        
        # Replace text
        updated_content = content.replace(old_text, new_text)
        
        # Write back to file
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(updated_content)
        
        replacements = content.count(old_text)
        
        log_operation(logger, "REPLACE_IN_FILE", {
            "filepath": str(file_path),
            "old_text_length": len(old_text),
            "new_text_length": len(new_text),
            "replacements_made": replacements
        })
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to replace text in file {filepath}: {e}")
        return False


def update_file_permissions(filepath: str, permissions: str) -> bool:
    """
    Update file permissions.
    
    Args:
        filepath: Path to the file
        permissions: Permissions in octal format (e.g., '755')
    
    Returns:
        True if successful, False otherwise
    """
    try:
        file_path = Path(filepath)
        
        if not file_path.exists():
            logger.warning(f"File {filepath} does not exist")
            return False
        
        # Convert permissions to integer
        perm_int = int(permissions, 8)
        
        # Change permissions
        file_path.chmod(perm_int)
        
        log_operation(logger, "UPDATE_PERMISSIONS", {
            "filepath": str(file_path),
            "permissions": permissions
        })
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to update permissions for {filepath}: {e}")
        return False
