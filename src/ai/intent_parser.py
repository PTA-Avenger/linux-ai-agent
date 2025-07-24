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
            
            # AI operations
            "ai_stats": [
                r"ai (?:stats|statistics)",
                r"show ai (?:stats|statistics)",
                r"ai agent (?:stats|statistics|status)",
            ],
            "ai_recommend": [
                r"ai recommend(?:ation)?s? (.+)",
                r"what (?:do you |would you )?(?:recommend|suggest) (?:for )?(.+)",
                r"ai (?:advice|suggestion)s? (?:for )?(.+)",
            ],
            "generate_command": [
                r"generate (?:a )?command (?:to |for |that )?(.+)",
                r"create (?:a )?command (?:to |for |that )?(.+)",
                r"make (?:a )?command (?:to |for |that )?(.+)",
            ],
            "generate_script": [
                r"generate (?:a )?script (?:to |for )?(.+)",
                r"create (?:a )?script (?:to |for )?(.+)",
                r"make (?:a )?script (?:to |for )?(.+)",
            ],
            "system_cleanup": [
                r"clean up (?:the )?system",
                r"cleanup system",
                r"clean up",
                r"system cleanup",
                r"clean (?:the )?system",
            ],
            "gemma_chat": [
                r"gemma (.+)",
                r"ask gemma (.+)",
                r"chat with gemma (.+)",
                r"gemma help (.+)",
            ],
            "ai_analyze": [
                r"ai analyze (.+)",
                r"analyze with ai (.+)",
                r"ai analysis (.+)",
                r"get ai analysis (.+)",
            ],
            "heuristic_scan": [
                r"heuristic scan (.+)",
                r"analyze (.+) heuristically",
                r"check (.+) with heuristics",
            ],
            "detailed_scan": [
                r"detailed scan (.+)",
                r"comprehensive scan (.+)",
                r"full report (.+)",
                r"detailed analysis (.+)",
                r"generate report (.+)",
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
        
        # Extract arguments from groups
        if match.groups():
            # First group is usually the main argument
            main_arg = match.group(1).strip()
            if main_arg:
                # For AI recommendations, treat as context
                if "recommend" in text or "suggest" in text or "advice" in text:
                    parameters["context"] = main_arg
                    parameters["path"] = main_arg  # Also set as path for compatibility
                # For command/script generation, treat as description
                elif "generate" in text or "create" in text:
                    parameters["description"] = main_arg
                    parameters["path"] = main_arg  # Also set as path for compatibility
                else:
                    # Default to path for file/directory operations
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
        Suggest commands based on partial input with fuzzy matching.
        
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
                "ai stats",
                "help"
            ]
        else:
            # Create example commands for each intent
            intent_examples = {
                "create_file": "create file <filename>",
                "read_file": "read file <filename>",
                "update_file": "update file <filename>",
                "delete_file": "delete file <filename>",
                "list_directory": "list directory <path>",
                "scan_file": "scan file <filename>",
                "scan_directory": "scan directory <path>",
                "quarantine_file": "quarantine file <filename>",
                "list_quarantine": "list quarantine",
                "restore_file": "restore file <filename>",
                "disk_usage": "check disk usage",
                "system_stats": "system stats",
                "monitor_directory": "monitor directory <path>",
                "heuristic_scan": "heuristic scan <filename>",
                "detailed_scan": "detailed scan <filename>",
                "system_cleanup": "clean up",
                "gemma_chat": "gemma <question>",
                "ai_analyze": "ai analyze <topic>",
                "ai_stats": "ai stats",
                "ai_recommend": "ai recommend <context>",
                "generate_command": "generate command <description>",
                "generate_script": "generate script <description>",
                "help": "help",
                "exit": "exit"
            }
            
            # Find fuzzy matches
            scored_suggestions = []
            for intent, example in intent_examples.items():
                score = self._calculate_fuzzy_score(partial_text, example.lower())
                if score > 0.3:  # Minimum similarity threshold
                    scored_suggestions.append((score, example))
            
            # Sort by score and return top suggestions
            scored_suggestions.sort(key=lambda x: x[0], reverse=True)
            suggestions = [suggestion for _, suggestion in scored_suggestions[:limit]]
        
        return suggestions
    
    def _calculate_fuzzy_score(self, input_text: str, target_text: str) -> float:
        """Calculate fuzzy similarity score between two strings."""
        # Simple fuzzy matching based on common substrings and character overlap
        input_words = set(input_text.split())
        target_words = set(target_text.split())
        
        # Word overlap score
        if target_words:
            word_overlap = len(input_words.intersection(target_words)) / len(target_words)
        else:
            word_overlap = 0
        
        # Character overlap score
        input_chars = set(input_text.replace(' ', ''))
        target_chars = set(target_text.replace(' ', ''))
        if target_chars:
            char_overlap = len(input_chars.intersection(target_chars)) / len(target_chars)
        else:
            char_overlap = 0
        
        # Substring score
        substring_score = 0
        for word in input_words:
            if any(word in target_word for target_word in target_words):
                substring_score += 0.5
        
        # Combine scores
        total_score = (word_overlap * 0.5 + char_overlap * 0.3 + substring_score * 0.2)
        return min(total_score, 1.0)
    
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
