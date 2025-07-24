"""
Command Generator for Linux AI Agent.
Generates shell scripts and commands from natural language descriptions.
"""

import os
import sys
import re
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utils import get_logger, log_operation

logger = get_logger("command_generator")


class CommandGenerator:
    """
    Generate shell commands and scripts from natural language descriptions.
    """
    
    def __init__(self):
        self.command_templates = {
            # File operations
            "backup": {
                "patterns": [
                    r"backup (?:the )?(.+?) (?:to |into )?(.+)",
                    r"create (?:a )?backup of (.+?) (?:in |to )?(.+)",
                    r"copy (.+?) to (.+?) as (?:a )?backup"
                ],
                "template": "cp -r '{source}' '{destination}/backup_$(date +%Y%m%d_%H%M%S)_{basename}'",
                "description": "Create a timestamped backup"
            },
            
            "archive": {
                "patterns": [
                    r"archive (.+?) (?:to |as )?(.+)",
                    r"compress (.+?) (?:into |to )?(.+)",
                    r"create (?:an )?archive of (.+?) (?:called |named )?(.+)"
                ],
                "template": "tar -czf '{destination}.tar.gz' '{source}'",
                "description": "Create compressed archive"
            },
            
            "find_files": {
                "patterns": [
                    r"find (?:all )?files (?:named |called )?(.+?) in (.+)",
                    r"search for (.+?) in (.+)",
                    r"locate files (?:matching )?(.+?) (?:in |under )?(.+)"
                ],
                "template": "find '{directory}' -name '{pattern}' -type f",
                "description": "Find files matching pattern"
            },
            
            "find_large_files": {
                "patterns": [
                    r"find large files in (.+)",
                    r"find files larger than (\d+)([MG]B?) in (.+)",
                    r"locate big files (?:in |under )?(.+)"
                ],
                "template": "find '{directory}' -type f -size +{size} -exec ls -lh {} \\; | sort -k5 -hr",
                "description": "Find large files"
            },
            
            # System monitoring
            "monitor_logs": {
                "patterns": [
                    r"monitor (?:the )?logs? (?:in |from )?(.+)",
                    r"watch (?:the )?log file (.+)",
                    r"tail (?:the )?logs? (?:in |from )?(.+)"
                ],
                "template": "tail -f '{logpath}'",
                "description": "Monitor log files in real-time"
            },
            
            "system_cleanup": {
                "patterns": [
                    r"clean up (?:the )?system",
                    r"free up disk space",
                    r"remove (?:temporary|temp) files"
                ],
                "template": "sudo apt autoremove -y && sudo apt autoclean && sudo journalctl --vacuum-time=7d",
                "description": "Clean up system and free disk space"
            },
            
            "check_disk_space": {
                "patterns": [
                    r"check disk space (?:in |for )?(.+)?",
                    r"show disk usage (?:of |for )?(.+)?",
                    r"how much space (?:is left |available )?(?:in |on )?(.+)?"
                ],
                "template": "df -h {path}",
                "description": "Check available disk space"
            },
            
            # Security operations
            "scan_directory": {
                "patterns": [
                    r"scan (.+?) for (?:malware|viruses?)",
                    r"check (.+?) for (?:threats|infections?)",
                    r"run (?:a )?security scan on (.+)"
                ],
                "template": "clamscan -r --infected --stdout '{directory}'",
                "description": "Scan directory for malware"
            },
            
            "check_permissions": {
                "patterns": [
                    r"check permissions (?:on |of |for )?(.+)",
                    r"show file permissions (?:for |of )?(.+)",
                    r"list permissions (?:in |of )?(.+)"
                ],
                "template": "ls -la '{path}'",
                "description": "Check file permissions"
            },
            
            # Process management
            "kill_process": {
                "patterns": [
                    r"kill (?:the )?process (?:named |called )?(.+)",
                    r"stop (?:the )?(.+?) process",
                    r"terminate (.+)"
                ],
                "template": "pkill -f '{process_name}'",
                "description": "Kill process by name"
            },
            
            "monitor_process": {
                "patterns": [
                    r"monitor (?:the )?process (?:named |called )?(.+)",
                    r"watch (?:the )?(.+?) process",
                    r"keep an eye on (.+)"
                ],
                "template": "watch -n 1 'ps aux | grep {process_name}'",
                "description": "Monitor process activity"
            },
            
            # Network operations
            "check_connectivity": {
                "patterns": [
                    r"check (?:internet )?connectivity (?:to )?(.+)?",
                    r"ping (.+)",
                    r"test connection (?:to )?(.+)"
                ],
                "template": "ping -c 4 '{host}'",
                "description": "Test network connectivity"
            },
            
            "download_file": {
                "patterns": [
                    r"download (.+?) (?:to |into )?(.+)?",
                    r"fetch (.+?) (?:and save (?:to |in )?(.+)?)?",
                    r"get (.+?) (?:from (?:the )?(?:internet|web))?"
                ],
                "template": "wget -O '{destination}' '{url}'",
                "description": "Download file from URL"
            }
        }
        
        # Common parameter patterns
        self.parameter_patterns = {
            "size": r"(\d+)([KMGT]?B?)",
            "time": r"(\d+)\s*(seconds?|minutes?|hours?|days?)",
            "url": r"https?://[^\s]+",
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "ip": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"
        }
        
        logger.info("Command Generator initialized")
    
    def generate_command(self, description: str) -> Dict[str, Any]:
        """
        Generate a shell command from natural language description.
        
        Args:
            description: Natural language description of desired command
            
        Returns:
            Dictionary with command details
        """
        description = description.strip().lower()
        
        if not description:
            return {
                "success": False,
                "error": "Empty description provided"
            }
        
        best_match = {
            "command_type": None,
            "confidence": 0.0,
            "parameters": {},
            "template": None
        }
        
        # Try to match against command templates
        for command_type, template_info in self.command_templates.items():
            for pattern in template_info["patterns"]:
                match = re.search(pattern, description, re.IGNORECASE)
                if match:
                    confidence = self._calculate_confidence(description, pattern, match)
                    
                    if confidence > best_match["confidence"]:
                        parameters = self._extract_parameters(match, command_type)
                        best_match = {
                            "command_type": command_type,
                            "confidence": confidence,
                            "parameters": parameters,
                            "template": template_info["template"],
                            "description": template_info["description"]
                        }
        
        if best_match["command_type"]:
            # Generate the actual command
            try:
                command = self._fill_template(
                    best_match["template"], 
                    best_match["parameters"]
                )
                
                result = {
                    "success": True,
                    "command": command,
                    "command_type": best_match["command_type"],
                    "confidence": best_match["confidence"],
                    "description": best_match["description"],
                    "parameters": best_match["parameters"],
                    "safety_level": self._assess_safety(command),
                    "requires_sudo": "sudo" in command
                }
                
                log_operation(logger, "GENERATE_COMMAND", {
                    "description": description,
                    "command_type": best_match["command_type"],
                    "confidence": best_match["confidence"]
                })
                
                return result
                
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error generating command: {str(e)}"
                }
        else:
            return {
                "success": False,
                "error": "Could not understand the command description",
                "suggestions": self._get_suggestions(description)
            }
    
    def _calculate_confidence(self, description: str, pattern: str, match: re.Match) -> float:
        """Calculate confidence score for pattern match."""
        # Base confidence from match coverage
        match_length = len(match.group(0))
        desc_length = len(description)
        coverage = match_length / desc_length
        
        # Boost for exact matches
        if match.group(0) == description:
            coverage = 1.0
        
        # Boost for capturing groups (parameters)
        if match.groups():
            coverage += 0.1
        
        # Penalize for very short matches
        if match_length < 5:
            coverage *= 0.5
        
        return min(coverage, 1.0)
    
    def _extract_parameters(self, match: re.Match, command_type: str) -> Dict[str, Any]:
        """Extract parameters from regex match."""
        parameters = {}
        groups = match.groups()
        
        if command_type == "backup":
            if len(groups) >= 2:
                parameters["source"] = groups[0].strip()
                parameters["destination"] = groups[1].strip()
                parameters["basename"] = Path(groups[0]).name
        
        elif command_type == "archive":
            if len(groups) >= 2:
                parameters["source"] = groups[0].strip()
                parameters["destination"] = groups[1].strip()
        
        elif command_type == "find_files":
            if len(groups) >= 2:
                parameters["pattern"] = groups[0].strip()
                parameters["directory"] = groups[1].strip()
        
        elif command_type == "find_large_files":
            if len(groups) >= 1:
                if len(groups) >= 3:  # Size specified
                    size_num = groups[0]
                    size_unit = groups[1].upper()
                    directory = groups[2]
                    parameters["size"] = f"{size_num}{size_unit}"
                    parameters["directory"] = directory.strip()
                else:  # Default size
                    parameters["size"] = "100M"
                    parameters["directory"] = groups[0].strip()
        
        elif command_type in ["monitor_logs", "scan_directory", "check_permissions"]:
            if len(groups) >= 1:
                parameters["logpath" if command_type == "monitor_logs" else 
                          "directory" if command_type == "scan_directory" else "path"] = groups[0].strip()
        
        elif command_type == "check_disk_space":
            if len(groups) >= 1 and groups[0]:
                parameters["path"] = groups[0].strip()
            else:
                parameters["path"] = ""
        
        elif command_type in ["kill_process", "monitor_process"]:
            if len(groups) >= 1:
                parameters["process_name"] = groups[0].strip()
        
        elif command_type == "check_connectivity":
            if len(groups) >= 1 and groups[0]:
                parameters["host"] = groups[0].strip()
            else:
                parameters["host"] = "8.8.8.8"  # Default to Google DNS
        
        elif command_type == "download_file":
            if len(groups) >= 1:
                parameters["url"] = groups[0].strip()
                if len(groups) >= 2 and groups[1]:
                    parameters["destination"] = groups[1].strip()
                else:
                    # Extract filename from URL
                    url_path = Path(groups[0])
                    parameters["destination"] = url_path.name or "downloaded_file"
        
        return parameters
    
    def _fill_template(self, template: str, parameters: Dict[str, Any]) -> str:
        """Fill command template with parameters."""
        command = template
        
        for key, value in parameters.items():
            placeholder = f"{{{key}}}"
            if placeholder in command:
                command = command.replace(placeholder, str(value))
        
        # Handle any remaining placeholders with defaults
        defaults = {
            "{path}": ".",
            "{directory}": ".",
            "{size}": "100M",
            "{host}": "8.8.8.8"
        }
        
        for placeholder, default in defaults.items():
            if placeholder in command:
                command = command.replace(placeholder, default)
        
        return command
    
    def _assess_safety(self, command: str) -> str:
        """Assess the safety level of a command."""
        dangerous_patterns = [
            r"rm\s+-rf\s+/",
            r"dd\s+.*of=/dev/",
            r"mkfs\.",
            r"fdisk",
            r"parted",
            r"shutdown",
            r"reboot",
            r"halt"
        ]
        
        moderate_patterns = [
            r"sudo",
            r"rm\s+",
            r"chmod\s+",
            r"chown\s+",
            r"mv\s+.*\s+/",
            r"cp\s+.*\s+/"
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return "dangerous"
        
        for pattern in moderate_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return "moderate"
        
        return "safe"
    
    def _get_suggestions(self, description: str) -> List[str]:
        """Get command suggestions based on partial description."""
        suggestions = []
        words = description.lower().split()
        
        # Look for key words and suggest related commands
        if any(word in words for word in ["backup", "copy", "save"]):
            suggestions.append("backup /path/to/source to /path/to/destination")
        
        if any(word in words for word in ["find", "search", "locate"]):
            suggestions.append("find files named pattern in /path/to/directory")
        
        if any(word in words for word in ["monitor", "watch", "tail"]):
            suggestions.append("monitor logs in /var/log/syslog")
        
        if any(word in words for word in ["clean", "cleanup", "free"]):
            suggestions.append("clean up the system")
        
        if any(word in words for word in ["scan", "check", "security"]):
            suggestions.append("scan /path/to/directory for malware")
        
        if any(word in words for word in ["disk", "space", "usage"]):
            suggestions.append("check disk space")
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def generate_script(self, description: str, commands: List[str]) -> Dict[str, Any]:
        """
        Generate a complete shell script from description and commands.
        
        Args:
            description: Description of what the script does
            commands: List of commands to include
            
        Returns:
            Dictionary with script details
        """
        if not commands:
            return {
                "success": False,
                "error": "No commands provided"
            }
        
        script_name = re.sub(r'[^a-zA-Z0-9_]', '_', description.lower())[:50]
        script_name = f"{script_name}.sh"
        
        script_content = f"""#!/bin/bash
# Script: {script_name}
# Description: {description}
# Generated by Linux AI Agent on $(date)

set -e  # Exit on any error

echo "Starting: {description}"

"""
        
        for i, command in enumerate(commands, 1):
            script_content += f"""
# Step {i}
echo "Step {i}: Executing command..."
{command}

"""
        
        script_content += """
echo "Script completed successfully!"
"""
        
        return {
            "success": True,
            "script_name": script_name,
            "script_content": script_content,
            "description": description,
            "command_count": len(commands),
            "safety_assessment": self._assess_script_safety(commands)
        }
    
    def _assess_script_safety(self, commands: List[str]) -> Dict[str, Any]:
        """Assess the overall safety of a script."""
        safety_levels = [self._assess_safety(cmd) for cmd in commands]
        
        dangerous_count = safety_levels.count("dangerous")
        moderate_count = safety_levels.count("moderate")
        safe_count = safety_levels.count("safe")
        
        if dangerous_count > 0:
            overall_safety = "dangerous"
        elif moderate_count > 0:
            overall_safety = "moderate"
        else:
            overall_safety = "safe"
        
        return {
            "overall_safety": overall_safety,
            "dangerous_commands": dangerous_count,
            "moderate_commands": moderate_count,
            "safe_commands": safe_count,
            "requires_review": dangerous_count > 0 or moderate_count > len(commands) // 2
        }
    
    def save_script(self, script_content: str, script_name: str, make_executable: bool = True) -> Dict[str, Any]:
        """
        Save a generated script to file.
        
        Args:
            script_content: Content of the script
            script_name: Name of the script file
            make_executable: Whether to make the script executable
            
        Returns:
            Dictionary with save results
        """
        try:
            script_path = Path(script_name)
            
            # Write script content
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make executable if requested
            if make_executable:
                script_path.chmod(0o755)
            
            return {
                "success": True,
                "script_path": str(script_path.absolute()),
                "executable": make_executable,
                "size": script_path.stat().st_size
            }
            
        except Exception as e:
            logger.error(f"Error saving script {script_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }