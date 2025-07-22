"""
Quarantine management for Linux AI Agent.
"""

import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional
import json
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utils import get_logger, log_operation


logger = get_logger("quarantine")


class QuarantineManager:
    """Manager for quarantining suspicious/malicious files."""
    
    def __init__(self, quarantine_dir: str = "quarantine"):
        self.quarantine_dir = Path(quarantine_dir)
        self.quarantine_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.quarantine_dir / "files").mkdir(exist_ok=True)
        (self.quarantine_dir / "metadata").mkdir(exist_ok=True)
        
        self.metadata_file = self.quarantine_dir / "quarantine_log.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load quarantine metadata from file."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {"quarantined_files": []}
        except Exception as e:
            logger.error(f"Error loading quarantine metadata: {e}")
            self.metadata = {"quarantined_files": []}
    
    def _save_metadata(self):
        """Save quarantine metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving quarantine metadata: {e}")
    
    def _generate_quarantine_filename(self, original_path: str) -> str:
        """Generate a safe filename for quarantined file."""
        original_name = Path(original_path).name
        timestamp = int(time.time())
        return f"{timestamp}_{original_name}"
    
    def quarantine_file(self, filepath: str, reason: str, scan_results: Optional[Dict] = None) -> Dict[str, any]:
        """
        Quarantine a file.
        
        Args:
            filepath: Path to the file to quarantine
            reason: Reason for quarantine
            scan_results: Optional scan results that triggered quarantine
        
        Returns:
            Dictionary with quarantine operation results
        """
        try:
            source_path = Path(filepath)
            
            if not source_path.exists():
                return {
                    "status": "error",
                    "message": f"File {filepath} does not exist"
                }
            
            if not source_path.is_file():
                return {
                    "status": "error",
                    "message": f"{filepath} is not a file"
                }
            
            # Generate quarantine filename
            quarantine_filename = self._generate_quarantine_filename(filepath)
            quarantine_path = self.quarantine_dir / "files" / quarantine_filename
            
            # Get file info before quarantine
            file_stat = source_path.stat()
            file_info = {
                "original_path": str(source_path),
                "quarantine_filename": quarantine_filename,
                "quarantine_path": str(quarantine_path),
                "size": file_stat.st_size,
                "modified_time": file_stat.st_mtime,
                "quarantine_time": time.time(),
                "reason": reason,
                "scan_results": scan_results or {}
            }
            
            # Move file to quarantine (safer than copy+delete for sensitive files)
            try:
                shutil.move(str(source_path), str(quarantine_path))
            except Exception as e:
                # If move fails, try copy and secure delete
                shutil.copy2(str(source_path), str(quarantine_path))
                self._secure_delete(source_path)
            
            # Update metadata
            self.metadata["quarantined_files"].append(file_info)
            self._save_metadata()
            
            # Save individual metadata file
            metadata_path = self.quarantine_dir / "metadata" / f"{quarantine_filename}.json"
            with open(metadata_path, 'w') as f:
                json.dump(file_info, f, indent=2)
            
            result = {
                "status": "success",
                "original_path": str(source_path),
                "quarantine_path": str(quarantine_path),
                "quarantine_filename": quarantine_filename,
                "reason": reason
            }
            
            log_operation(logger, "QUARANTINE_FILE", {
                "original_path": str(source_path),
                "quarantine_path": str(quarantine_path),
                "reason": reason
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error quarantining file {filepath}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _secure_delete(self, filepath: Path, passes: int = 3):
        """Securely delete a file by overwriting."""
        try:
            if not filepath.exists():
                return
            
            size = filepath.stat().st_size
            
            with open(filepath, 'r+b') as f:
                for _ in range(passes):
                    f.seek(0)
                    f.write(os.urandom(size))
                    f.flush()
                    os.fsync(f.fileno())
            
            filepath.unlink()
            
        except Exception as e:
            logger.error(f"Error securely deleting {filepath}: {e}")
            # Fallback to regular delete
            try:
                filepath.unlink()
            except Exception:
                pass
    
    def list_quarantined_files(self) -> List[Dict[str, any]]:
        """
        List all quarantined files.
        
        Returns:
            List of quarantined file information
        """
        return self.metadata.get("quarantined_files", [])
    
    def get_quarantine_info(self, quarantine_filename: str) -> Optional[Dict[str, any]]:
        """
        Get information about a quarantined file.
        
        Args:
            quarantine_filename: Name of the quarantined file
        
        Returns:
            File information dictionary, None if not found
        """
        for file_info in self.metadata["quarantined_files"]:
            if file_info["quarantine_filename"] == quarantine_filename:
                return file_info
        return None
    
    def restore_file(self, quarantine_filename: str, restore_path: Optional[str] = None) -> Dict[str, any]:
        """
        Restore a quarantined file.
        
        Args:
            quarantine_filename: Name of the quarantined file
            restore_path: Optional custom restore path
        
        Returns:
            Dictionary with restore operation results
        """
        try:
            # Find file info
            file_info = self.get_quarantine_info(quarantine_filename)
            if not file_info:
                return {
                    "status": "error",
                    "message": f"Quarantined file {quarantine_filename} not found"
                }
            
            quarantine_path = Path(file_info["quarantine_path"])
            if not quarantine_path.exists():
                return {
                    "status": "error",
                    "message": f"Quarantined file {quarantine_path} does not exist"
                }
            
            # Determine restore path
            if restore_path:
                target_path = Path(restore_path)
            else:
                target_path = Path(file_info["original_path"])
            
            # Ensure target directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Restore file
            shutil.move(str(quarantine_path), str(target_path))
            
            # Remove from quarantine metadata
            self.metadata["quarantined_files"] = [
                f for f in self.metadata["quarantined_files"]
                if f["quarantine_filename"] != quarantine_filename
            ]
            self._save_metadata()
            
            # Remove individual metadata file
            metadata_path = self.quarantine_dir / "metadata" / f"{quarantine_filename}.json"
            if metadata_path.exists():
                metadata_path.unlink()
            
            result = {
                "status": "success",
                "quarantine_filename": quarantine_filename,
                "restored_path": str(target_path)
            }
            
            log_operation(logger, "RESTORE_FILE", {
                "quarantine_filename": quarantine_filename,
                "restored_path": str(target_path)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error restoring file {quarantine_filename}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def delete_quarantined_file(self, quarantine_filename: str) -> Dict[str, any]:
        """
        Permanently delete a quarantined file.
        
        Args:
            quarantine_filename: Name of the quarantined file
        
        Returns:
            Dictionary with delete operation results
        """
        try:
            # Find file info
            file_info = self.get_quarantine_info(quarantine_filename)
            if not file_info:
                return {
                    "status": "error",
                    "message": f"Quarantined file {quarantine_filename} not found"
                }
            
            quarantine_path = Path(file_info["quarantine_path"])
            
            # Securely delete the file
            if quarantine_path.exists():
                self._secure_delete(quarantine_path)
            
            # Remove from quarantine metadata
            self.metadata["quarantined_files"] = [
                f for f in self.metadata["quarantined_files"]
                if f["quarantine_filename"] != quarantine_filename
            ]
            self._save_metadata()
            
            # Remove individual metadata file
            metadata_path = self.quarantine_dir / "metadata" / f"{quarantine_filename}.json"
            if metadata_path.exists():
                metadata_path.unlink()
            
            result = {
                "status": "success",
                "quarantine_filename": quarantine_filename,
                "message": "File permanently deleted"
            }
            
            log_operation(logger, "DELETE_QUARANTINED", {
                "quarantine_filename": quarantine_filename
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error deleting quarantined file {quarantine_filename}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def cleanup_old_quarantine(self, days: int = 30) -> Dict[str, any]:
        """
        Clean up quarantined files older than specified days.
        
        Args:
            days: Age threshold in days
        
        Returns:
            Dictionary with cleanup results
        """
        try:
            cutoff_time = time.time() - (days * 24 * 60 * 60)
            files_to_remove = []
            
            for file_info in self.metadata["quarantined_files"]:
                if file_info.get("quarantine_time", 0) < cutoff_time:
                    files_to_remove.append(file_info["quarantine_filename"])
            
            removed_count = 0
            for filename in files_to_remove:
                result = self.delete_quarantined_file(filename)
                if result["status"] == "success":
                    removed_count += 1
            
            result = {
                "status": "success",
                "removed_files": removed_count,
                "total_candidates": len(files_to_remove),
                "cutoff_days": days
            }
            
            log_operation(logger, "CLEANUP_QUARANTINE", {
                "removed_files": removed_count,
                "cutoff_days": days
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error cleaning up quarantine: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_quarantine_stats(self) -> Dict[str, any]:
        """
        Get quarantine statistics.
        
        Returns:
            Dictionary with quarantine statistics
        """
        try:
            files = self.metadata["quarantined_files"]
            
            if not files:
                return {
                    "total_files": 0,
                    "total_size_mb": 0.0,
                    "oldest_quarantine": None,
                    "newest_quarantine": None
                }
            
            total_size = sum(f.get("size", 0) for f in files)
            quarantine_times = [f.get("quarantine_time", 0) for f in files if f.get("quarantine_time")]
            
            stats = {
                "total_files": len(files),
                "total_size_mb": total_size / (1024 * 1024),
                "oldest_quarantine": min(quarantine_times) if quarantine_times else None,
                "newest_quarantine": max(quarantine_times) if quarantine_times else None
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting quarantine stats: {e}")
            return {}
