"""
Heuristic scanning using entropy analysis for Linux AI Agent.
"""

import os
import math
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import Counter
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utils import get_logger, log_operation


logger = get_logger("heuristic_scanner")


class HeuristicScanner:
    """Heuristic scanner using entropy analysis and other techniques."""
    
    def __init__(self):
        self.high_entropy_threshold = 7.5
        self.suspicious_extensions = {
            '.exe', '.scr', '.bat', '.cmd', '.com', '.pif', '.vbs', '.js',
            '.jar', '.sh', '.py', '.pl', '.rb', '.php'
        }
        self.suspicious_names = {
            'autorun.inf', 'desktop.ini', 'thumbs.db', '.htaccess'
        }
        self.max_file_size = 100 * 1024 * 1024  # 100MB
    
    def calculate_entropy(self, data: bytes) -> float:
        """
        Calculate Shannon entropy of data.
        
        Args:
            data: Bytes data to analyze
        
        Returns:
            Entropy value (0-8, higher = more random)
        """
        if not data:
            return 0.0
        
        # Count frequency of each byte value
        byte_counts = Counter(data)
        data_len = len(data)
        
        # Calculate entropy
        entropy = 0.0
        for count in byte_counts.values():
            probability = count / data_len
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def analyze_file_entropy(self, filepath: str, chunk_size: int = 8192) -> Dict[str, any]:
        """
        Analyze file entropy in chunks.
        
        Args:
            filepath: Path to the file to analyze
            chunk_size: Size of chunks to analyze
        
        Returns:
            Dictionary with entropy analysis results
        """
        try:
            file_path = Path(filepath)
            
            if not file_path.exists():
                return {
                    "status": "error",
                    "message": f"File {filepath} does not exist"
                }
            
            if not file_path.is_file():
                return {
                    "status": "error",
                    "message": f"{filepath} is not a file"
                }
            
            file_size = file_path.stat().st_size
            
            if file_size == 0:
                return {
                    "status": "clean",
                    "file_size": 0,
                    "average_entropy": 0.0,
                    "max_entropy": 0.0,
                    "high_entropy_chunks": 0
                }
            
            if file_size > self.max_file_size:
                return {
                    "status": "skipped",
                    "message": f"File too large ({file_size} bytes)"
                }
            
            entropies = []
            high_entropy_chunks = 0
            
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    entropy = self.calculate_entropy(chunk)
                    entropies.append(entropy)
                    
                    if entropy > self.high_entropy_threshold:
                        high_entropy_chunks += 1
            
            if not entropies:
                return {
                    "status": "clean",
                    "file_size": file_size,
                    "average_entropy": 0.0,
                    "max_entropy": 0.0,
                    "high_entropy_chunks": 0
                }
            
            average_entropy = sum(entropies) / len(entropies)
            max_entropy = max(entropies)
            
            # Determine suspiciousness
            suspicious = False
            reasons = []
            
            if average_entropy > self.high_entropy_threshold:
                suspicious = True
                reasons.append(f"High average entropy: {average_entropy:.2f}")
            
            if high_entropy_chunks > len(entropies) * 0.5:  # More than 50% high entropy chunks
                suspicious = True
                reasons.append(f"Many high entropy chunks: {high_entropy_chunks}/{len(entropies)}")
            
            result = {
                "filepath": str(file_path),
                "file_size": file_size,
                "total_chunks": len(entropies),
                "average_entropy": average_entropy,
                "max_entropy": max_entropy,
                "high_entropy_chunks": high_entropy_chunks,
                "suspicious": suspicious,
                "reasons": reasons,
                "status": "suspicious" if suspicious else "clean"
            }
            
            log_operation(logger, "ANALYZE_ENTROPY", {
                "filepath": str(file_path),
                "average_entropy": average_entropy,
                "suspicious": suspicious
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing entropy for {filepath}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def check_suspicious_attributes(self, filepath: str) -> Dict[str, any]:
        """
        Check for suspicious file attributes.
        
        Args:
            filepath: Path to the file to check
        
        Returns:
            Dictionary with attribute analysis results
        """
        try:
            file_path = Path(filepath)
            
            if not file_path.exists():
                return {
                    "status": "error",
                    "message": f"File {filepath} does not exist"
                }
            
            suspicious = False
            reasons = []
            
            # Check file extension
            extension = file_path.suffix.lower()
            if extension in self.suspicious_extensions:
                suspicious = True
                reasons.append(f"Suspicious extension: {extension}")
            
            # Check file name
            filename = file_path.name.lower()
            if filename in self.suspicious_names:
                suspicious = True
                reasons.append(f"Suspicious filename: {filename}")
            
            # Check if hidden file (starts with .)
            if file_path.name.startswith('.') and file_path.name not in {'.', '..'}:
                reasons.append("Hidden file")
            
            # Check file permissions
            try:
                stat = file_path.stat()
                mode = stat.st_mode
                
                # Check if executable
                if os.access(file_path, os.X_OK):
                    reasons.append("Executable file")
                
                # Check unusual permissions
                perms = oct(mode)[-3:]
                if perms in ['777', '666']:
                    suspicious = True
                    reasons.append(f"Unusual permissions: {perms}")
                
            except (OSError, FileNotFoundError):
                pass
            
            # Check file size anomalies
            try:
                size = file_path.stat().st_size
                if size == 0:
                    reasons.append("Zero-byte file")
                elif size > 100 * 1024 * 1024:  # > 100MB
                    reasons.append("Very large file")
            except (OSError, FileNotFoundError):
                pass
            
            result = {
                "filepath": str(file_path),
                "suspicious": suspicious,
                "reasons": reasons,
                "status": "suspicious" if suspicious else "clean"
            }
            
            log_operation(logger, "CHECK_ATTRIBUTES", {
                "filepath": str(file_path),
                "suspicious": suspicious,
                "reason_count": len(reasons)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking attributes for {filepath}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def scan_file(self, filepath: str) -> Dict[str, any]:
        """
        Perform complete heuristic scan of a file.
        
        Args:
            filepath: Path to the file to scan
        
        Returns:
            Dictionary with complete scan results
        """
        try:
            file_path = Path(filepath)
            
            if not file_path.exists():
                return {
                    "status": "error",
                    "message": f"File {filepath} does not exist"
                }
            
            # Perform entropy analysis
            entropy_result = self.analyze_file_entropy(filepath)
            
            # Check suspicious attributes
            attr_result = self.check_suspicious_attributes(filepath)
            
            # Combine results
            suspicious = entropy_result.get("suspicious", False) or attr_result.get("suspicious", False)
            
            reasons = []
            if entropy_result.get("reasons"):
                reasons.extend(entropy_result["reasons"])
            if attr_result.get("reasons"):
                reasons.extend(attr_result["reasons"])
            
            result = {
                "filepath": str(file_path),
                "entropy_analysis": entropy_result,
                "attribute_analysis": attr_result,
                "overall_suspicious": suspicious,
                "all_reasons": reasons,
                "status": "suspicious" if suspicious else "clean",
                "scan_type": "heuristic"
            }
            
            # Calculate risk score (0-100)
            risk_score = 0
            if entropy_result.get("average_entropy", 0) > self.high_entropy_threshold:
                risk_score += 40
            if entropy_result.get("high_entropy_chunks", 0) > 0:
                risk_score += 20
            if attr_result.get("suspicious", False):
                risk_score += 30
            if len(reasons) > 3:
                risk_score += 10
            
            result["risk_score"] = min(risk_score, 100)
            
            log_operation(logger, "HEURISTIC_SCAN", {
                "filepath": str(file_path),
                "suspicious": suspicious,
                "risk_score": result["risk_score"]
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in heuristic scan for {filepath}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def scan_directory(self, directory: str, recursive: bool = True) -> Dict[str, any]:
        """
        Perform heuristic scan of a directory.
        
        Args:
            directory: Path to the directory to scan
            recursive: Whether to scan recursively
        
        Returns:
            Dictionary with directory scan results
        """
        try:
            dir_path = Path(directory)
            
            if not dir_path.exists():
                return {
                    "status": "error",
                    "message": f"Directory {directory} does not exist"
                }
            
            if not dir_path.is_dir():
                return {
                    "status": "error",
                    "message": f"{directory} is not a directory"
                }
            
            scanned_files = []
            suspicious_files = []
            total_files = 0
            
            # Get files to scan
            if recursive:
                files = [f for f in dir_path.rglob('*') if f.is_file()]
            else:
                files = [f for f in dir_path.iterdir() if f.is_file()]
            
            for file_path in files:
                total_files += 1
                
                # Skip very large files
                try:
                    if file_path.stat().st_size > self.max_file_size:
                        continue
                except (OSError, FileNotFoundError):
                    continue
                
                scan_result = self.scan_file(str(file_path))
                scanned_files.append(scan_result)
                
                if scan_result.get("overall_suspicious", False):
                    suspicious_files.append(scan_result)
            
            result = {
                "directory": str(dir_path),
                "recursive": recursive,
                "total_files": total_files,
                "scanned_files": len(scanned_files),
                "suspicious_files": len(suspicious_files),
                "suspicious_file_details": suspicious_files,
                "status": "suspicious" if suspicious_files else "clean"
            }
            
            log_operation(logger, "HEURISTIC_SCAN_DIRECTORY", {
                "directory": str(dir_path),
                "total_files": total_files,
                "suspicious_files": len(suspicious_files)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in heuristic directory scan for {directory}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def generate_file_hash(self, filepath: str, algorithm: str = "sha256") -> Optional[str]:
        """
        Generate hash of a file.
        
        Args:
            filepath: Path to the file
            algorithm: Hash algorithm (md5, sha1, sha256)
        
        Returns:
            File hash as hex string, None if failed
        """
        try:
            file_path = Path(filepath)
            
            if not file_path.exists() or not file_path.is_file():
                return None
            
            hash_obj = hashlib.new(algorithm)
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            logger.error(f"Error generating hash for {filepath}: {e}")
            return None
