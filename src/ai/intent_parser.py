"""
Intent parser for Linux AI Agent.
"""

import os
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utils import get_logger, log_operation


logger = get_logger("intent_parser")


class IntentParser:
    """Simple intent parser for natural language commands."""
    
    def __init__(self):
        self.intent_patterns = {
            # File operations
            "create_file": [
                r"create (?:a )?file (?:called |named )?(.+)",
                r"make (?:a )?new file (.+)",
                r"touch (.+)",
            ],
            "read_file": [
                r"read (?:the )?file (.+)",
                r"show (?:me )?(?:the )?contents? of (.+)",
                r"cat (.+)",
                r"display (.+)",
            ],
            "update_file": [
                r"update (?:the )?file (.+)",
                r"modify (.+)",
                r"edit (.+)",
                r"change (.+)",
            ],
            "delete_file": [
                r"delete (?:the )?file (.+)",
                r"remove (.+)",
                r"rm (.+)",
            ],
            
            # Directory operations
            "create_directory": [
                r"create (?:a )?(?:directory|folder) (?:called |named )?(.+)",
                r"make (?:a )?(?:directory|folder) (.+)",
                r"mkdir (.+)",
            ],
            "list_directory": [
                r"list (?:the )?(?:contents of |files in )?(?:directory |folder )?(.+)",
                r"ls (.+)",
                r"show (?:me )?(?:what's in |files in )(.+)",
            ],
            
            # Scanning operations
            "scan_file": [
                r"scan (?:the )?file (.+)",
                r"check (?:the )?file (.+) for (?:viruses|malware)",
                r"analyze (.+)",
            ],
            "scan_directory": [
                r"scan (?:the )?(?:directory|folder) (.+)",
                r"check (.+) for (?:viruses|malware)",
                r"scan (.+) recursively",
            ],
            
            # System monitoring
            "disk_usage": [
                r"(?:check|show) disk usage",
                r"how much (?:disk )?space",
                r"disk space",
                r"df",
            ],
            "system_stats": [
                r"(?:show|check) system (?:stats|statistics)",
                r"system (?:info|information)",
                r"system status",
            ],
            "monitor_directory": [
                r"monitor (?:the )?(?:directory|folder) (.+)",
                r"watch (.+)",
                r"track changes in (.+)",
            ],
            
            # Quarantine operations
            "quarantine_file": [
                r"quarantine (?:the )?file (.+)",
                r"isolate (.+)",
                r"move (.+) to quarantine",
            ],
            "list_quarantine": [
                r"(?:list|show) quarantined files",
                r"what's in quarantine",
                r"quarantine (?:list|status)",
            ],
            "restore_file": [
                r"restore (?:the )?file (.+)",
                r"unquarantine (.+)",
                r"bring back (.+)",
            ],
            
            # General commands
            "help": [
                r"help",
                r"what can you do",
                r"(?:show )?commands",
                r"usage",
            ],
            "exit": [
                r"exit",
                r"quit",
                r"bye",
                r"goodbye",
            ],
        }
        
        # Common parameters
        self.parameter_patterns = {
            "recursive": r"(?:recursively|recursive|-r)",
            "force": r"(?:force|forced|-f)",
            "verbose": r"(?:verbose|-v)",
            "quiet": r"(?:quiet|silent|-q)",
        }
    
    def parse_intent(self, text: str) -> Dict[str, any]:
        """
        Parse user input to extract intent and parameters.
        
        Args:
            text: User input text
        
        Returns:
            Dictionary with parsed intent information
        """
        text = text.strip().lower()
        
        if not text:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "parameters": {},
                "original_text": text
            }
        
        best_match = {
            "intent": "unknown",
            "confidence": 0.0,
            "parameters": {},
            "matched_pattern": None,
            "extracted_args": []
        }
        
        # Try to match against intent patterns
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    confidence = self._calculate_confidence(text, pattern, match)
                    
                    if confidence > best_match["confidence"]:
                        best_match = {
                            "intent": intent,
                            "confidence": confidence,
                            "parameters": self._extract_parameters(text, match),
                            "matched_pattern": pattern,
                            "extracted_args": list(match.groups())
                        }
        
        # Extract common parameters
        for param, pattern in self.parameter_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                best_match["parameters"][param] = True
        
        result = {
            "intent": best_match["intent"],
            "confidence": best_match["confidence"],
            "parameters": best_match["parameters"],
            "original_text": text,
            "matched_pattern": best_match["matched_pattern"],
            "extracted_args": best_match["extracted_args"]
        }
        
        log_operation(logger, "PARSE_INTENT", {
            "original_text": text,
            "intent": result["intent"],
            "confidence": result["confidence"]
        })
        
        return result
    
    def _calculate_confidence(self, text: str, pattern: str, match: re.Match) -> float:
        """Calculate confidence score for a pattern match."""
        # Base confidence from match length vs text length
        match_length = len(match.group(0))
        text_length = len(text)
        base_confidence = match_length / text_length
        
        # Boost confidence for exact matches
        if match.group(0) == text:
            base_confidence = 1.0
        
        # Boost confidence for patterns that capture arguments
        if match.groups():
            base_confidence += 0.1
        
        # Penalize if there's a lot of unmatched text
        unmatched_ratio = (text_length - match_length) / text_length
        if unmatched_ratio > 0.5:
            base_confidence *= 0.7
        
        return min(base_confidence, 1.0)
    
    def _extract_parameters(self, text: str, match: re.Match) -> Dict[str, any]:
        """Extract parameters from matched text."""
        parameters = {}
        
        # Extract file/directory paths from groups
        if match.groups():
            # First group is usually the main argument (file/directory path)
            main_arg = match.group(1).strip()
            if main_arg:
                parameters["path"] = main_arg
                
                # Try to determine if it's a file or directory
                path_obj = Path(main_arg)
                if path_obj.exists():
                    parameters["is_file"] = path_obj.is_file()
                    parameters["is_directory"] = path_obj.is_dir()
                else:
                    # Guess based on extension
                    if path_obj.suffix:
                        parameters["is_file"] = True
                        parameters["is_directory"] = False
                    else:
                        # Could be either, let the handler decide
                        parameters["is_file"] = None
                        parameters["is_directory"] = None
        
        return parameters
    
    def suggest_commands(self, partial_text: str, limit: int = 5) -> List[str]:
        """
        Suggest commands based on partial input.
        
        Args:
            partial_text: Partial user input
            limit: Maximum number of suggestions
        
        Returns:
            List of suggested commands
        """
        partial_text = partial_text.strip().lower()
        suggestions = []
        
        if not partial_text:
            # Return common commands
            suggestions = [
                "scan file <filename>",
                "create file <filename>",
                "read file <filename>",
                "list directory <path>",
                "check disk usage",
                "help"
            ]
        else:
            # Find matching patterns
            for intent, patterns in self.intent_patterns.items():
                for pattern in patterns:
                    # Convert regex pattern to readable format
                    readable = self._pattern_to_readable(pattern)
                    if partial_text in readable.lower():
                        suggestions.append(readable)
                        if len(suggestions) >= limit:
                            break
                if len(suggestions) >= limit:
                    break
        
        return suggestions[:limit]
    
    def _pattern_to_readable(self, pattern: str) -> str:
        """Convert regex pattern to human-readable format."""
        # Simple conversion for common patterns
        readable = pattern
        readable = re.sub(r'\(\?\:', '', readable)
        readable = re.sub(r'\)\?', '', readable)
        readable = re.sub(r'\(\.\+\)', '<argument>', readable)
        readable = re.sub(r'\\', '', readable)
        readable = re.sub(r'\|', ' or ', readable)
        return readable
    
    def get_intent_help(self, intent: str) -> Dict[str, any]:
        """
        Get help information for a specific intent.
        
        Args:
            intent: Intent name
        
        Returns:
            Dictionary with help information
        """
        help_info = {
            "create_file": {
                "description": "Create a new file",
                "examples": ["create file test.txt", "make new file document.md"],
                "parameters": ["path"]
            },
            "read_file": {
                "description": "Read contents of a file",
                "examples": ["read file config.txt", "show contents of data.json"],
                "parameters": ["path"]
            },
            "scan_file": {
                "description": "Scan a file for malware",
                "examples": ["scan file suspicious.exe", "check file download.zip for viruses"],
                "parameters": ["path"]
            },
            "disk_usage": {
                "description": "Check disk usage statistics",
                "examples": ["check disk usage", "show disk space"],
                "parameters": []
            },
            # Add more help info as needed
        }
        
        return help_info.get(intent, {
            "description": f"Execute {intent} operation",
            "examples": [],
            "parameters": []
        })
    
    def validate_parameters(self, intent: str, parameters: Dict[str, any]) -> Dict[str, any]:
        """
        Validate parameters for a given intent.
        
        Args:
            intent: Intent name
            parameters: Parameters to validate
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Define required parameters for each intent
        required_params = {
            "create_file": ["path"],
            "read_file": ["path"],
            "update_file": ["path"],
            "delete_file": ["path"],
            "scan_file": ["path"],
            "scan_directory": ["path"],
            "quarantine_file": ["path"],
            "restore_file": ["path"],
        }
        
        # Check required parameters
        if intent in required_params:
            for param in required_params[intent]:
                if param not in parameters or not parameters[param]:
                    validation["valid"] = False
                    validation["errors"].append(f"Missing required parameter: {param}")
        
        # Validate file/directory paths
        if "path" in parameters and parameters["path"]:
            path = parameters["path"]
            
            # Check for suspicious characters (basic validation)
            if any(char in path for char in ['..', '|', ';', '&']):
                validation["warnings"].append("Path contains potentially unsafe characters")
            
            # Check if path exists for operations that require it
            if intent in ["read_file", "update_file", "delete_file", "scan_file"]:
                if not Path(path).exists():
                    validation["warnings"].append(f"Path does not exist: {path}")
        
        return validation
