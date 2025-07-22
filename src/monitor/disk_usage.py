"""
Disk usage monitoring for Linux AI Agent.
"""

import os
import psutil
import time
from pathlib import Path
from typing import Dict, List, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utils import get_logger, log_operation


logger = get_logger("monitor_disk")


def get_disk_usage(path: str = "/") -> Dict[str, float]:
    """
    Get disk usage statistics for a given path.
    
    Args:
        path: Path to check disk usage for
    
    Returns:
        Dictionary with usage statistics
    """
    try:
        usage = psutil.disk_usage(path)
        
        stats = {
            "total_gb": usage.total / (1024**3),
            "used_gb": usage.used / (1024**3),
            "free_gb": usage.free / (1024**3),
            "used_percent": (usage.used / usage.total) * 100
        }
        
        log_operation(logger, "GET_DISK_USAGE", {
            "path": path,
            "used_percent": stats["used_percent"],
            "free_gb": stats["free_gb"]
        })
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get disk usage for {path}: {e}")
        return {}


def get_directory_size(directory: str) -> Dict[str, float]:
    """
    Get the total size of a directory and its contents.
    
    Args:
        directory: Path to the directory
    
    Returns:
        Dictionary with size information
    """
    try:
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.warning(f"Directory {directory} does not exist")
            return {}
        
        total_size = 0
        file_count = 0
        
        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                try:
                    total_size += file_path.stat().st_size
                    file_count += 1
                except (OSError, FileNotFoundError):
                    # Skip files that can't be accessed
                    continue
        
        stats = {
            "total_bytes": total_size,
            "total_mb": total_size / (1024**2),
            "total_gb": total_size / (1024**3),
            "file_count": file_count
        }
        
        log_operation(logger, "GET_DIRECTORY_SIZE", {
            "directory": directory,
            "total_mb": stats["total_mb"],
            "file_count": file_count
        })
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get directory size for {directory}: {e}")
        return {}


def get_largest_files(directory: str, limit: int = 10) -> List[Dict[str, any]]:
    """
    Get the largest files in a directory.
    
    Args:
        directory: Path to the directory
        limit: Number of files to return
    
    Returns:
        List of file information dictionaries
    """
    try:
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.warning(f"Directory {directory} does not exist")
            return []
        
        files_info = []
        
        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    files_info.append({
                        "path": str(file_path),
                        "size_bytes": stat.st_size,
                        "size_mb": stat.st_size / (1024**2),
                        "modified_time": stat.st_mtime
                    })
                except (OSError, FileNotFoundError):
                    continue
        
        # Sort by size (descending) and take top N
        largest_files = sorted(files_info, key=lambda x: x["size_bytes"], reverse=True)[:limit]
        
        log_operation(logger, "GET_LARGEST_FILES", {
            "directory": directory,
            "limit": limit,
            "files_found": len(files_info)
        })
        
        return largest_files
        
    except Exception as e:
        logger.error(f"Failed to get largest files in {directory}: {e}")
        return []


def monitor_disk_space(path: str = "/", threshold_percent: float = 90.0) -> Dict[str, any]:
    """
    Monitor disk space and return status.
    
    Args:
        path: Path to monitor
        threshold_percent: Warning threshold percentage
    
    Returns:
        Dictionary with monitoring status
    """
    try:
        usage = get_disk_usage(path)
        
        if not usage:
            return {"status": "error", "message": "Failed to get disk usage"}
        
        status = {
            "path": path,
            "usage": usage,
            "threshold_percent": threshold_percent,
            "status": "ok"
        }
        
        if usage["used_percent"] >= threshold_percent:
            status["status"] = "warning"
            status["message"] = f"Disk usage {usage['used_percent']:.1f}% exceeds threshold {threshold_percent}%"
            logger.warning(status["message"])
        else:
            status["message"] = f"Disk usage {usage['used_percent']:.1f}% is within threshold"
        
        log_operation(logger, "MONITOR_DISK_SPACE", {
            "path": path,
            "used_percent": usage["used_percent"],
            "status": status["status"]
        })
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to monitor disk space for {path}: {e}")
        return {"status": "error", "message": str(e)}


class FileActivityHandler(FileSystemEventHandler):
    """Handler for file system events."""
    
    def __init__(self):
        self.events = []
        self.max_events = 1000  # Limit stored events
    
    def on_any_event(self, event):
        if len(self.events) >= self.max_events:
            self.events.pop(0)  # Remove oldest event
        
        self.events.append({
            "timestamp": time.time(),
            "event_type": event.event_type,
            "src_path": event.src_path,
            "is_directory": event.is_directory
        })
    
    def get_recent_events(self, seconds: int = 60) -> List[Dict[str, any]]:
        """Get events from the last N seconds."""
        cutoff_time = time.time() - seconds
        return [event for event in self.events if event["timestamp"] >= cutoff_time]


def get_file_activity(directory: str, duration_seconds: int = 60) -> List[Dict[str, any]]:
    """
    Monitor file activity in a directory for a specified duration.
    
    Args:
        directory: Directory to monitor
        duration_seconds: How long to monitor (in seconds)
    
    Returns:
        List of file activity events
    """
    try:
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.warning(f"Directory {directory} does not exist")
            return []
        
        event_handler = FileActivityHandler()
        observer = Observer()
        observer.schedule(event_handler, str(dir_path), recursive=True)
        
        observer.start()
        time.sleep(duration_seconds)
        observer.stop()
        observer.join()
        
        events = event_handler.get_recent_events(duration_seconds)
        
        log_operation(logger, "GET_FILE_ACTIVITY", {
            "directory": directory,
            "duration_seconds": duration_seconds,
            "events_count": len(events)
        })
        
        return events
        
    except Exception as e:
        logger.error(f"Failed to monitor file activity in {directory}: {e}")
        return []


def get_system_stats() -> Dict[str, any]:
    """
    Get general system statistics.
    
    Returns:
        Dictionary with system statistics
    """
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Disk usage for root
        disk = psutil.disk_usage('/')
        
        # Load average (Linux only)
        load_avg = None
        try:
            load_avg = os.getloadavg()
        except (OSError, AttributeError):
            pass
        
        stats = {
            "cpu_percent": cpu_percent,
            "memory": {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_percent": memory.percent
            },
            "disk": {
                "total_gb": disk.total / (1024**3),
                "free_gb": disk.free / (1024**3),
                "used_percent": (disk.used / disk.total) * 100
            },
            "load_average": load_avg,
            "timestamp": time.time()
        }
        
        log_operation(logger, "GET_SYSTEM_STATS", {
            "cpu_percent": cpu_percent,
            "memory_used_percent": memory.percent,
            "disk_used_percent": stats["disk"]["used_percent"]
        })
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        return {}
