"""
ClamAV wrapper for Linux AI Agent.
Enhanced with auto-fallback to heuristic scanning.
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
    """Enhanced wrapper for ClamAV antivirus scanner with heuristic fallback."""
    
    def __init__(self, enable_fallback: bool = True):
        self.clamscan_path = self._find_clamscan()
        self.available = self.clamscan_path is not None
        self.enable_fallback = enable_fallback
        self.heuristic_scanner = None
        
        if not self.available:
            logger.warning("ClamAV not found. Install with: sudo apt install clamav clamav-daemon")
            if self.enable_fallback:
                self._initialize_fallback()
        else:
            logger.info(f"ClamAV found at: {self.clamscan_path}")
    
    def _initialize_fallback(self):
        """Initialize heuristic scanner as fallback."""
        try:
            from .heuristics import HeuristicScanner
            self.heuristic_scanner = HeuristicScanner()
            logger.info("Heuristic scanner initialized as ClamAV fallback")
        except ImportError as e:
            logger.error(f"Could not initialize heuristic fallback: {e}")
    
    def _find_clamscan(self) -> Optional[str]:
        """Find the clamscan executable with enhanced detection."""
        common_paths = [
            "/usr/bin/clamscan",
            "/usr/local/bin/clamscan",
            "/opt/clamav/bin/clamscan",
            "/snap/bin/clamscan"  # Snap package location
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
        
        # Try whereis command
        try:
            result = subprocess.run(["whereis", "clamscan"], capture_output=True, text=True)
            if result.returncode == 0:
                parts = result.stdout.split()
                for part in parts[1:]:  # Skip the first part which is "clamscan:"
                    if os.path.exists(part) and os.access(part, os.X_OK):
                        return part
        except Exception:
            pass
        
        return None
    
    def update_database(self) -> bool:
        """Update ClamAV virus database with improved error handling."""
        if not self.available:
            logger.error("ClamAV not available for database update")
            return False
        
        try:
            # Try freshclam first
            result = subprocess.run(
                ["sudo", "freshclam"],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout for database updates
            )
            
            success = result.returncode == 0
            
            # If sudo freshclam fails, try without sudo (for user installations)
            if not success:
                logger.info("Trying freshclam without sudo...")
                result = subprocess.run(
                    ["freshclam"],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                success = result.returncode == 0
            
            log_operation(logger, "UPDATE_DATABASE", {
                "success": success,
                "return_code": result.returncode,
                "stderr": result.stderr[:500] if result.stderr else None,
                "stdout": result.stdout[:500] if result.stdout else None
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
        Scan a single file with ClamAV or fallback to heuristic scanning.
        
        Args:
            filepath: Path to the file to scan
        
        Returns:
            Dictionary with scan results
        """
        if not self.available:
            if self.enable_fallback and self.heuristic_scanner:
                logger.info(f"ClamAV not available, using heuristic fallback for {filepath}")
                heuristic_result = self.heuristic_scanner.scan_file(filepath)
                
                # Convert heuristic result to ClamAV-compatible format
                return {
                    "status": "suspicious" if heuristic_result.get("overall_suspicious", False) else "clean",
                    "infected": False,  # Heuristic doesn't detect known viruses
                    "virus_name": None,
                    "scan_type": "heuristic_fallback",
                    "heuristic_details": heuristic_result,
                    "message": "Scanned using heuristic analysis (ClamAV not available)"
                }
            else:
                return {
                    "status": "error",
                    "message": "ClamAV not available and no fallback configured",
                    "infected": False,
                    "virus_name": None
                }
        
        try:
            file_path = Path(filepath)
            
            if not file_path.exists():
                return {
                    "status": "error",
                    "message": f"File {filepath} does not exist",
                    "infected": False,
                    "virus_name": None
                }
            
            if not file_path.is_file():
                return {
                    "status": "error", 
                    "message": f"{filepath} is not a file",
                    "infected": False,
                    "virus_name": None
                }
            
            # Run ClamAV scan with enhanced options
            cmd = [
                self.clamscan_path,
                "--no-summary",      # Don't show summary
                "--infected",        # Only show infected files
                "--stdout",          # Output to stdout
                "--recursive=no",    # Don't scan recursively for single file
                str(file_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minutes timeout for single file
            )
            
            # Parse ClamAV output
            scan_result = self._parse_clamscan_output(result, filepath)
            
            log_operation(logger, "SCAN_FILE", {
                "filepath": str(file_path),
                "status": scan_result["status"],
                "infected": scan_result.get("infected", False),
                "return_code": result.returncode
            })
            
            return scan_result
            
        except subprocess.TimeoutExpired:
            logger.error(f"ClamAV scan timeout for file: {filepath}")
            return {
                "status": "error",
                "message": "Scan timeout",
                "infected": False,
                "virus_name": None
            }
        except Exception as e:
            logger.error(f"Error scanning file {filepath}: {e}")
            
            # Try fallback on error
            if self.enable_fallback and self.heuristic_scanner:
                logger.info(f"ClamAV error, trying heuristic fallback for {filepath}")
                heuristic_result = self.heuristic_scanner.scan_file(filepath)
                
                return {
                    "status": "suspicious" if heuristic_result.get("overall_suspicious", False) else "clean",
                    "infected": False,
                    "virus_name": None,
                    "scan_type": "heuristic_fallback_error",
                    "heuristic_details": heuristic_result,
                    "message": f"ClamAV error, used heuristic fallback: {str(e)}"
                }
            
            return {
                "status": "error",
                "message": str(e),
                "infected": False,
                "virus_name": None
            }
    
    def _parse_clamscan_output(self, result: subprocess.CompletedProcess, filepath: str) -> Dict[str, any]:
        """Parse ClamAV scan output with enhanced detection."""
        output = result.stdout.strip()
        stderr = result.stderr.strip()
        
        # ClamAV return codes:
        # 0 = clean
        # 1 = infected
        # 2 = error
        
        if result.returncode == 0:
            return {
                "status": "clean",
                "infected": False,
                "virus_name": None,
                "scan_type": "clamav",
                "message": "File is clean"
            }
        elif result.returncode == 1:
            # Parse virus name from output
            virus_name = "Unknown"
            if output:
                # ClamAV output format: "filepath: VirusName FOUND"
                lines = output.split('\n')
                for line in lines:
                    if 'FOUND' in line:
                        parts = line.split(':')
                        if len(parts) >= 2:
                            virus_part = parts[1].strip()
                            if virus_part.endswith(' FOUND'):
                                virus_name = virus_part[:-6].strip()  # Remove ' FOUND'
                        break
            
            return {
                "status": "infected",
                "infected": True,
                "virus_name": virus_name,
                "scan_type": "clamav",
                "message": f"Infected with {virus_name}"
            }
        else:
            # Error occurred
            error_msg = stderr if stderr else "Unknown ClamAV error"
            return {
                "status": "error",
                "infected": False,
                "virus_name": None,
                "scan_type": "clamav",
                "message": error_msg
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
