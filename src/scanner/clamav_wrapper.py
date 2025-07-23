"""
ClamAV wrapper for Linux AI Agent.
"""

import subprocess
import os
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utils import get_logger, log_operation


logger = get_logger("clamav_scanner")


class ClamAVScanner:
    """Wrapper for ClamAV antivirus scanner."""
    
    def __init__(self):
        self.clamscan_path = self._find_clamscan()
        self.available = self.clamscan_path is not None
        
        if not self.available:
            logger.warning("ClamAV not found. Install with: sudo apt install clamav clamav-daemon")
    
    def _find_clamscan(self) -> Optional[str]:
        """Find the clamscan executable."""
        common_paths = [
            "/usr/bin/clamscan",
            "/usr/local/bin/clamscan",
            "/opt/clamav/bin/clamscan"
        ]
        
        for path in common_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path
        
        # Try to find in PATH
        try:
            result = subprocess.run(["which", "clamscan"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        
        return None
    
    def update_database(self) -> bool:
        """Update ClamAV virus database."""
        if not self.available:
            logger.error("ClamAV not available")
            return False
        
        try:
            # Try freshclam first
            result = subprocess.run(
                ["sudo", "freshclam"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            success = result.returncode == 0
            
            log_operation(logger, "UPDATE_DATABASE", {
                "success": success,
                "return_code": result.returncode,
                "stderr": result.stderr[:500] if result.stderr else None
            })
            
            if success:
                logger.info("ClamAV database updated successfully")
            else:
                logger.error(f"Failed to update ClamAV database: {result.stderr}")
            
            return success
            
        except subprocess.TimeoutExpired:
            logger.error("ClamAV database update timed out")
            return False
        except Exception as e:
            logger.error(f"Error updating ClamAV database: {e}")
            return False
    
    def scan_file(self, filepath: str) -> Dict[str, any]:
        """
        Scan a single file with ClamAV.
        
        Args:
            filepath: Path to the file to scan
        
        Returns:
            Dictionary with scan results
        """
        if not self.available:
            return {
                "status": "error",
                "message": "ClamAV not available",
                "infected": False
            }
        
        try:
            file_path = Path(filepath)
            
            if not file_path.exists():
                return {
                    "status": "error",
                    "message": f"File {filepath} does not exist",
                    "infected": False
                }
            
            # Run clamscan
            cmd = [
                self.clamscan_path,
                "--no-summary",
                "--infected",
                str(file_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout per file
            )
            
            # ClamAV return codes:
            # 0 - No virus found
            # 1 - Virus found
            # 2 - Some error occurred
            
            scan_result = {
                "filepath": str(file_path),
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "infected": result.returncode == 1,
                "status": "clean"
            }
            
            if result.returncode == 1:
                scan_result["status"] = "infected"
                # Extract virus name from output
                if "FOUND" in result.stdout:
                    virus_line = [line for line in result.stdout.split('\n') if "FOUND" in line]
                    if virus_line:
                        scan_result["virus_name"] = virus_line[0].split(": ")[1].replace(" FOUND", "")
            elif result.returncode == 2:
                scan_result["status"] = "error"
                scan_result["message"] = result.stderr
            
            log_operation(logger, "SCAN_FILE", {
                "filepath": str(file_path),
                "status": scan_result["status"],
                "infected": scan_result["infected"]
            })
            
            return scan_result
            
        except subprocess.TimeoutExpired:
            logger.error(f"ClamAV scan timed out for {filepath}")
            return {
                "status": "error",
                "message": "Scan timed out",
                "infected": False
            }
        except Exception as e:
            logger.error(f"Error scanning file {filepath}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "infected": False
            }
    
    def scan_directory(self, directory: str, recursive: bool = True) -> Dict[str, any]:
        """
        Scan a directory with ClamAV.
        
        Args:
            directory: Path to the directory to scan
            recursive: Whether to scan recursively
        
        Returns:
            Dictionary with scan results
        """
        if not self.available:
            return {
                "status": "error",
                "message": "ClamAV not available",
                "infected_files": []
            }
        
        try:
            dir_path = Path(directory)
            
            if not dir_path.exists():
                return {
                    "status": "error",
                    "message": f"Directory {directory} does not exist",
                    "infected_files": []
                }
            
            # Build command
            cmd = [
                self.clamscan_path,
                "--infected",
                "--bell"
            ]
            
            if recursive:
                cmd.append("--recursive")
            
            cmd.append(str(dir_path))
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout for directory scan
            )
            
            # Parse output for infected files
            infected_files = []
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if "FOUND" in line:
                        parts = line.split(": ")
                        if len(parts) >= 2:
                            filepath = parts[0]
                            virus_name = parts[1].replace(" FOUND", "")
                            infected_files.append({
                                "filepath": filepath,
                                "virus_name": virus_name
                            })
            
            scan_result = {
                "directory": str(dir_path),
                "recursive": recursive,
                "return_code": result.returncode,
                "infected_files": infected_files,
                "total_infected": len(infected_files),
                "status": "clean" if result.returncode == 0 else "infected" if result.returncode == 1 else "error"
            }
            
            if result.returncode == 2:
                scan_result["error_message"] = result.stderr
            
            log_operation(logger, "SCAN_DIRECTORY", {
                "directory": str(dir_path),
                "recursive": recursive,
                "total_infected": len(infected_files),
                "status": scan_result["status"]
            })
            
            return scan_result
            
        except subprocess.TimeoutExpired:
            logger.error(f"ClamAV directory scan timed out for {directory}")
            return {
                "status": "error",
                "message": "Scan timed out",
                "infected_files": []
            }
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "infected_files": []
            }
    
    def get_version(self) -> Optional[str]:
        """Get ClamAV version."""
        if not self.available:
            return None
        
        try:
            result = subprocess.run(
                [self.clamscan_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            
        except Exception as e:
            logger.error(f"Error getting ClamAV version: {e}")
        
        return None
    
    def is_available(self) -> bool:
        """Check if ClamAV is available."""
        return self.available
