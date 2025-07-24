"""
Command Line Interface for Linux AI Agent.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from crud import create_file, read_file, update_file, delete_file
from monitor import get_disk_usage, monitor_disk_space, get_file_activity
from scanner import ClamAVScanner, HeuristicScanner, QuarantineManager
from ai import IntentParser, RLAgent, CommandGenerator
from utils import get_logger, log_operation

try:
    from colorama import init, Fore, Back, Style
    init()  # Initialize colorama
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False


logger = get_logger("cli")


class CLI:
    """Command Line Interface for Linux AI Agent."""
    
    def __init__(self):
        self.intent_parser = IntentParser()
        self.rl_agent = RLAgent()
        self.command_generator = CommandGenerator()
        self.clamav_scanner = ClamAVScanner()
        self.heuristic_scanner = HeuristicScanner()
        self.quarantine_manager = QuarantineManager()
        
        self.running = False
        self.command_history = []
        
        # Colors
        if COLORS_AVAILABLE:
            self.colors = {
                'header': Fore.CYAN + Style.BRIGHT,
                'success': Fore.GREEN,
                'warning': Fore.YELLOW,
                'error': Fore.RED,
                'info': Fore.BLUE,
                'reset': Style.RESET_ALL,
                'bold': Style.BRIGHT
            }
        else:
            self.colors = {key: '' for key in ['header', 'success', 'warning', 'error', 'info', 'reset', 'bold']}
        
        logger.info("CLI initialized")
    
    def print_colored(self, text: str, color: str = 'reset'):
        """Print colored text."""
        print(f"{self.colors.get(color, '')}{text}{self.colors['reset']}")
    
    def print_banner(self):
        """Print application banner."""
        banner = """
🛡️  Linux AI Agent
═══════════════════════════════════════════════════════════════
A modular Python-based AI agent for Linux file operations,
system monitoring, malware detection, and heuristic scanning.
═══════════════════════════════════════════════════════════════
        """
        self.print_colored(banner, 'header')
    
    def print_help(self):
        """Print help information."""
        help_text = """
Available Commands:
═══════════════════

📁 File Operations:
  • create file <path>          - Create a new file
  • read file <path>            - Read file contents
  • update file <path>          - Update file contents
  • delete file <path>          - Delete a file
  • list directory <path>       - List directory contents

🔍 Scanning Operations:
  • scan file <path>            - Scan file for malware
  • scan directory <path>       - Scan directory for malware
  • heuristic scan <path>       - Perform heuristic analysis

🛡️ Quarantine Operations:
  • quarantine file <path>      - Move file to quarantine
  • list quarantine             - Show quarantined files
  • restore file <name>         - Restore quarantined file
  • delete quarantine <name>    - Permanently delete quarantined file

📊 System Monitoring:
  • disk usage                  - Check disk usage
  • system stats                - Show system statistics
  • monitor directory <path>    - Monitor directory activity

🤖 AI Operations:
  • ai recommend <context>      - Get AI recommendations
  • ai stats                    - Show AI agent statistics
  • generate command <desc>     - Generate shell command from description
  • generate script <desc>      - Generate shell script from description

⚙️ General:
  • help                        - Show this help
  • exit                        - Exit the application

Examples:
  scan file /home/user/download.zip
  create file test.txt
  quarantine file suspicious.exe
  disk usage
  generate command backup /var/log to /backup
  generate script that cleans up system daily
        """
        self.print_colored(help_text, 'info')
    
    def execute_command(self, command: str) -> bool:
        """
        Execute a parsed command.
        
        Args:
            command: User command string
        
        Returns:
            True to continue, False to exit
        """
        if not command.strip():
            return True
        
        # Add to history
        self.command_history.append(command)
        
        # Parse intent
        intent_result = self.intent_parser.parse_intent(command)
        intent = intent_result["intent"]
        parameters = intent_result["parameters"]
        
        if intent_result["confidence"] < 0.3:
            self.print_colored(f"❓ I'm not sure what you mean by '{command}'", 'warning')
            self.print_colored("Type 'help' to see available commands.", 'info')
            return True
        
        try:
            # Execute based on intent
            if intent == "help":
                self.print_help()
            
            elif intent == "exit":
                self.print_colored("👋 Goodbye!", 'success')
                return False
            
            elif intent == "create_file":
                self._handle_create_file(parameters)
            
            elif intent == "read_file":
                self._handle_read_file(parameters)
            
            elif intent == "update_file":
                self._handle_update_file(parameters)
            
            elif intent == "delete_file":
                self._handle_delete_file(parameters)
            
            elif intent == "list_directory":
                self._handle_list_directory(parameters)
            
            elif intent == "scan_file":
                self._handle_scan_file(parameters)
            
            elif intent == "scan_directory":
                self._handle_scan_directory(parameters)
            
            elif intent == "quarantine_file":
                self._handle_quarantine_file(parameters)
            
            elif intent == "list_quarantine":
                self._handle_list_quarantine()
            
            elif intent == "restore_file":
                self._handle_restore_file(parameters)
            
            elif intent == "disk_usage":
                self._handle_disk_usage()
            
            elif intent == "system_stats":
                self._handle_system_stats()
            
            elif intent == "monitor_directory":
                self._handle_monitor_directory(parameters)
            
            elif intent == "heuristic_scan":
                self._handle_heuristic_scan(parameters)
            
            elif intent == "generate_command":
                self._handle_generate_command(parameters, command)
            
            elif intent == "generate_script":
                self._handle_generate_script(parameters, command)
            
            else:
                self.print_colored(f"❓ Command '{intent}' is not implemented yet.", 'warning')
        
        except Exception as e:
            self.print_colored(f"❌ Error executing command: {e}", 'error')
            logger.error(f"Error executing command '{command}': {e}")
        
        return True
    
    def _handle_create_file(self, parameters: Dict[str, Any]):
        """Handle file creation."""
        if "path" not in parameters:
            self.print_colored("❌ Please specify a file path.", 'error')
            return
        
        filepath = parameters["path"]
        content = input(f"Enter content for {filepath} (or press Enter for empty file): ")
        
        success = create_file(filepath, content)
        if success:
            self.print_colored(f"✅ File '{filepath}' created successfully.", 'success')
        else:
            self.print_colored(f"❌ Failed to create file '{filepath}'.", 'error')
    
    def _handle_read_file(self, parameters: Dict[str, Any]):
        """Handle file reading."""
        if "path" not in parameters:
            self.print_colored("❌ Please specify a file path.", 'error')
            return
        
        filepath = parameters["path"]
        content = read_file(filepath)
        
        if content is not None:
            self.print_colored(f"📄 Contents of '{filepath}':", 'info')
            print("-" * 50)
            print(content)
            print("-" * 50)
        else:
            self.print_colored(f"❌ Failed to read file '{filepath}'.", 'error')
    
    def _handle_update_file(self, parameters: Dict[str, Any]):
        """Handle file updating."""
        if "path" not in parameters:
            self.print_colored("❌ Please specify a file path.", 'error')
            return
        
        filepath = parameters["path"]
        
        # Read current content
        current_content = read_file(filepath)
        if current_content is None:
            self.print_colored(f"❌ Cannot read file '{filepath}'.", 'error')
            return
        
        self.print_colored(f"📄 Current content of '{filepath}':", 'info')
        print(current_content[:200] + "..." if len(current_content) > 200 else current_content)
        
        new_content = input("Enter new content: ")
        success = update_file(filepath, new_content)
        
        if success:
            self.print_colored(f"✅ File '{filepath}' updated successfully.", 'success')
        else:
            self.print_colored(f"❌ Failed to update file '{filepath}'.", 'error')
    
    def _handle_delete_file(self, parameters: Dict[str, Any]):
        """Handle file deletion."""
        if "path" not in parameters:
            self.print_colored("❌ Please specify a file path.", 'error')
            return
        
        filepath = parameters["path"]
        
        # Confirm deletion
        confirm = input(f"⚠️  Are you sure you want to delete '{filepath}'? (y/N): ")
        if confirm.lower() != 'y':
            self.print_colored("❌ Deletion cancelled.", 'warning')
            return
        
        success = delete_file(filepath)
        if success:
            self.print_colored(f"✅ File '{filepath}' deleted successfully.", 'success')
        else:
            self.print_colored(f"❌ Failed to delete file '{filepath}'.", 'error')
    
    def _handle_list_directory(self, parameters: Dict[str, Any]):
        """Handle directory listing."""
        if "path" not in parameters:
            path = "."
        else:
            path = parameters["path"]
        
        from crud.read import list_directory
        items = list_directory(path)
        
        if items:
            self.print_colored(f"📁 Contents of '{path}':", 'info')
            for item in items[:20]:  # Limit to first 20 items
                item_path = Path(item)
                if item_path.is_file():
                    self.print_colored(f"  📄 {item_path.name}", 'reset')
                else:
                    self.print_colored(f"  📁 {item_path.name}/", 'info')
            
            if len(items) > 20:
                self.print_colored(f"  ... and {len(items) - 20} more items", 'warning')
        else:
            self.print_colored(f"❌ Failed to list directory '{path}' or directory is empty.", 'error')
    
    def _handle_scan_file(self, parameters: Dict[str, Any]):
        """Handle file scanning."""
        if "path" not in parameters:
            self.print_colored("❌ Please specify a file path.", 'error')
            return
        
        filepath = parameters["path"]
        self.print_colored(f"🔍 Scanning file '{filepath}'...", 'info')
        
        # ClamAV scan
        clamav_result = self.clamav_scanner.scan_file(filepath)
        
        # Heuristic scan
        heuristic_result = self.heuristic_scanner.scan_file(filepath)
        
        # Display results
        self.print_colored("📊 Scan Results:", 'header')
        
        # ClamAV results
        if clamav_result["status"] == "infected":
            self.print_colored(f"🦠 ClamAV: INFECTED - {clamav_result.get('virus_name', 'Unknown')}", 'error')
        elif clamav_result["status"] == "clean":
            self.print_colored("✅ ClamAV: Clean", 'success')
        else:
            self.print_colored(f"⚠️  ClamAV: {clamav_result.get('message', 'Error')}", 'warning')
        
        # Heuristic results
        if heuristic_result.get("overall_suspicious", False):
            risk_score = heuristic_result.get("risk_score", 0)
            self.print_colored(f"🔍 Heuristic: SUSPICIOUS (Risk: {risk_score}%)", 'warning')
            for reason in heuristic_result.get("all_reasons", []):
                self.print_colored(f"    • {reason}", 'warning')
        else:
            self.print_colored("✅ Heuristic: Clean", 'success')
        
        # AI recommendation
        context = {
            "file_size": Path(filepath).stat().st_size if Path(filepath).exists() else 0,
            "scan_results": {
                "infected": clamav_result.get("infected", False),
                "suspicious": heuristic_result.get("overall_suspicious", False)
            }
        }
        
        recommendations = self.rl_agent.get_recommendations(context)
        if recommendations:
            self.print_colored("🤖 AI Recommendations:", 'header')
            for rec in recommendations[:2]:
                confidence = int(rec["confidence"] * 100)
                self.print_colored(f"    • {rec['action']} (confidence: {confidence}%)", 'info')
    
    def _handle_heuristic_scan(self, parameters: Dict[str, Any]):
        """Handle dedicated heuristic scanning."""
        if "path" not in parameters:
            self.print_colored("❌ Please specify a file path for heuristic scan.", 'error')
            return
        
        filepath = parameters["path"]
        self.print_colored(f"🔍 Performing heuristic scan on '{filepath}'...", 'info')
        
        # Heuristic scan only
        heuristic_result = self.heuristic_scanner.scan_file(filepath)
        
        # Display detailed heuristic results
        self.print_colored("📊 Heuristic Analysis Results:", 'header')
        
        if heuristic_result.get("overall_suspicious", False):
            risk_score = heuristic_result.get("risk_score", 0)
            self.print_colored(f"⚠️  Status: SUSPICIOUS (Risk Score: {risk_score}%)", 'warning')
            
            # Show detailed analysis
            entropy_analysis = heuristic_result.get("entropy_analysis", {})
            if entropy_analysis:
                avg_entropy = entropy_analysis.get("average_entropy", 0)
                max_entropy = entropy_analysis.get("max_entropy", 0)
                high_entropy_chunks = entropy_analysis.get("high_entropy_chunks", 0)
                
                self.print_colored(f"📈 Entropy Analysis:", 'info')
                self.print_colored(f"    • Average entropy: {avg_entropy:.2f}", 'info')
                self.print_colored(f"    • Maximum entropy: {max_entropy:.2f}", 'info')
                self.print_colored(f"    • High entropy chunks: {high_entropy_chunks}", 'info')
            
            # Show all reasons
            for reason in heuristic_result.get("all_reasons", []):
                self.print_colored(f"    • {reason}", 'warning')
                
            # AI recommendation for heuristic results
            context = {
                "file_size": Path(filepath).stat().st_size if Path(filepath).exists() else 0,
                "scan_results": {
                    "infected": False,
                    "suspicious": True,
                    "risk_score": risk_score
                }
            }
            
            recommendations = self.rl_agent.get_recommendations(context)
            if recommendations:
                self.print_colored("🤖 AI Recommendations:", 'header')
                for rec in recommendations[:3]:  # Show more recommendations for heuristic
                    confidence = int(rec["confidence"] * 100)
                    self.print_colored(f"    • {rec['action']} (confidence: {confidence}%)", 'info')
        else:
            self.print_colored("✅ Status: Clean", 'success')
            entropy_analysis = heuristic_result.get("entropy_analysis", {})
            if entropy_analysis:
                avg_entropy = entropy_analysis.get("average_entropy", 0)
                self.print_colored(f"📈 Average entropy: {avg_entropy:.2f} (within normal range)", 'success')
    
    def _handle_scan_directory(self, parameters: Dict[str, Any]):
        """Handle directory scanning."""
        if "path" not in parameters:
            self.print_colored("❌ Please specify a directory path.", 'error')
            return
        
        dirpath = parameters["path"]
        recursive = parameters.get("recursive", True)
        
        self.print_colored(f"🔍 Scanning directory '{dirpath}' {'recursively' if recursive else 'non-recursively'}...", 'info')
        
        # Heuristic scan (ClamAV directory scan can be very slow)
        heuristic_result = self.heuristic_scanner.scan_directory(dirpath, recursive)
        
        # Display results
        self.print_colored("📊 Directory Scan Results:", 'header')
        
        total_files = heuristic_result.get("total_files", 0)
        suspicious_files = heuristic_result.get("suspicious_files", 0)
        
        self.print_colored(f"📁 Total files: {total_files}", 'info')
        self.print_colored(f"🔍 Scanned files: {heuristic_result.get('scanned_files', 0)}", 'info')
        
        if suspicious_files > 0:
            self.print_colored(f"⚠️  Suspicious files: {suspicious_files}", 'warning')
            
            # Show details of suspicious files
            for file_info in heuristic_result.get("suspicious_file_details", [])[:5]:
                filepath = file_info.get("filepath", "Unknown")
                risk_score = file_info.get("risk_score", 0)
                self.print_colored(f"    • {filepath} (Risk: {risk_score}%)", 'warning')
            
            if suspicious_files > 5:
                self.print_colored(f"    ... and {suspicious_files - 5} more", 'warning')
        else:
            self.print_colored("✅ No suspicious files found", 'success')
    
    def _handle_quarantine_file(self, parameters: Dict[str, Any]):
        """Handle file quarantine."""
        if "path" not in parameters:
            self.print_colored("❌ Please specify a file path.", 'error')
            return
        
        filepath = parameters["path"]
        reason = input(f"Enter reason for quarantining '{filepath}': ") or "User requested"
        
        result = self.quarantine_manager.quarantine_file(filepath, reason)
        
        if result["status"] == "success":
            self.print_colored(f"✅ File '{filepath}' quarantined successfully.", 'success')
            self.print_colored(f"📁 Quarantine location: {result['quarantine_filename']}", 'info')
        else:
            self.print_colored(f"❌ Failed to quarantine file: {result.get('message', 'Unknown error')}", 'error')
    
    def _handle_list_quarantine(self):
        """Handle quarantine listing."""
        quarantined_files = self.quarantine_manager.list_quarantined_files()
        
        if not quarantined_files:
            self.print_colored("✅ No files in quarantine.", 'success')
            return
        
        self.print_colored(f"🛡️ Quarantined Files ({len(quarantined_files)}):", 'header')
        
        for file_info in quarantined_files:
            original_path = file_info.get("original_path", "Unknown")
            quarantine_filename = file_info.get("quarantine_filename", "Unknown")
            reason = file_info.get("reason", "No reason provided")
            size_mb = file_info.get("size", 0) / (1024 * 1024)
            
            self.print_colored(f"📄 {quarantine_filename}", 'warning')
            self.print_colored(f"    Original: {original_path}", 'reset')
            self.print_colored(f"    Reason: {reason}", 'reset')
            self.print_colored(f"    Size: {size_mb:.2f} MB", 'reset')
            print()
    
    def _handle_restore_file(self, parameters: Dict[str, Any]):
        """Handle file restoration."""
        if "path" not in parameters:
            self.print_colored("❌ Please specify a quarantine filename.", 'error')
            return
        
        quarantine_filename = parameters["path"]
        
        result = self.quarantine_manager.restore_file(quarantine_filename)
        
        if result["status"] == "success":
            self.print_colored(f"✅ File restored to: {result['restored_path']}", 'success')
        else:
            self.print_colored(f"❌ Failed to restore file: {result.get('message', 'Unknown error')}", 'error')
    
    def _handle_disk_usage(self):
        """Handle disk usage check."""
        self.print_colored("💾 Checking disk usage...", 'info')
        
        usage = get_disk_usage("/")
        
        if usage:
            self.print_colored("�� Disk Usage Statistics:", 'header')
            self.print_colored(f"    Total: {usage['total_gb']:.2f} GB", 'info')
            self.print_colored(f"    Used: {usage['used_gb']:.2f} GB ({usage['used_percent']:.1f}%)", 'info')
            self.print_colored(f"    Free: {usage['free_gb']:.2f} GB", 'info')
            
            if usage['used_percent'] > 90:
                self.print_colored("⚠️  Warning: Disk usage is very high!", 'warning')
            elif usage['used_percent'] > 80:
                self.print_colored("⚠️  Warning: Disk usage is high.", 'warning')
        else:
            self.print_colored("❌ Failed to get disk usage information.", 'error')
    
    def _handle_system_stats(self):
        """Handle system statistics."""
        from monitor.disk_usage import get_system_stats
        
        self.print_colored("📊 Getting system statistics...", 'info')
        
        stats = get_system_stats()
        
        if stats:
            self.print_colored("📊 System Statistics:", 'header')
            self.print_colored(f"    CPU Usage: {stats.get('cpu_percent', 0):.1f}%", 'info')
            
            memory = stats.get('memory', {})
            self.print_colored(f"    Memory: {memory.get('used_percent', 0):.1f}% used", 'info')
            self.print_colored(f"            {memory.get('available_gb', 0):.2f} GB available", 'info')
            
            disk = stats.get('disk', {})
            self.print_colored(f"    Disk: {disk.get('used_percent', 0):.1f}% used", 'info')
            self.print_colored(f"          {disk.get('free_gb', 0):.2f} GB free", 'info')
            
            load_avg = stats.get('load_average')
            if load_avg:
                self.print_colored(f"    Load Average: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}", 'info')
        else:
            self.print_colored("❌ Failed to get system statistics.", 'error')
    
    def _handle_monitor_directory(self, parameters: Dict[str, Any]):
        """Handle directory monitoring."""
        if "path" not in parameters:
            self.print_colored("❌ Please specify a directory path.", 'error')
            return
        
        dirpath = parameters["path"]
        duration = 30  # 30 seconds
        
        self.print_colored(f"👀 Monitoring '{dirpath}' for {duration} seconds...", 'info')
        self.print_colored("Press Ctrl+C to stop early.", 'warning')
        
        try:
            events = get_file_activity(dirpath, duration)
            
            if events:
                self.print_colored(f"📊 File Activity ({len(events)} events):", 'header')
                for event in events[-10:]:  # Show last 10 events
                    event_type = event.get("event_type", "unknown")
                    src_path = event.get("src_path", "unknown")
                    is_dir = "📁" if event.get("is_directory", False) else "📄"
                    
                    self.print_colored(f"    {is_dir} {event_type}: {src_path}", 'info')
                
                if len(events) > 10:
                    self.print_colored(f"    ... and {len(events) - 10} more events", 'warning')
            else:
                self.print_colored("✅ No file activity detected.", 'success')
                
        except KeyboardInterrupt:
            self.print_colored("\n⏹️  Monitoring stopped by user.", 'warning')
    
    def _handle_generate_command(self, parameters: Dict[str, Any], original_command: str):
        """Handle command generation from natural language."""
        # Extract the description from the original command
        description = original_command.replace("generate command", "").replace("create command", "").strip()
        
        if not description:
            description = input("Enter command description: ")
        
        if not description:
            self.print_colored("❌ Please provide a description for command generation.", 'error')
            return
        
        self.print_colored(f"🤖 Generating command: {description}", 'info')
        
        result = self.command_generator.generate_command(description)
        
        if result.get("success"):
            command = result["command"]
            confidence = result["confidence"]
            safety_level = result["safety_level"]
            
            self.print_colored("✅ Generated Command:", 'success')
            self.print_colored(f"📝 Command: {command}", 'info')
            self.print_colored(f"🎯 Confidence: {confidence:.2f}", 'info')
            self.print_colored(f"🛡️  Safety Level: {safety_level}", 
                              'warning' if safety_level != 'safe' else 'success')
            
            if result.get("requires_sudo"):
                self.print_colored("⚠️  Note: This command requires sudo privileges", 'warning')
            
            # Ask if user wants to execute the command
            if safety_level != 'dangerous':
                execute = input("Execute this command? (y/N): ").lower().strip()
                if execute == 'y':
                    try:
                        import subprocess
                        result = subprocess.run(command, shell=True, capture_output=True, text=True)
                        if result.returncode == 0:
                            self.print_colored("✅ Command executed successfully:", 'success')
                            if result.stdout:
                                print(result.stdout)
                        else:
                            self.print_colored("❌ Command failed:", 'error')
                            if result.stderr:
                                print(result.stderr)
                    except Exception as e:
                        self.print_colored(f"❌ Error executing command: {e}", 'error')
            else:
                self.print_colored("⚠️  Command marked as dangerous - execution blocked for safety", 'error')
                
        else:
            self.print_colored(f"❌ Failed to generate command: {result.get('error', 'Unknown error')}", 'error')
            suggestions = result.get("suggestions", [])
            if suggestions:
                self.print_colored("💡 Suggestions:", 'info')
                for suggestion in suggestions:
                    self.print_colored(f"    • {suggestion}", 'info')
    
    def _handle_generate_script(self, parameters: Dict[str, Any], original_command: str):
        """Handle script generation from natural language."""
        # Extract the description from the original command
        description = original_command.replace("generate script", "").replace("create script", "").strip()
        
        if not description:
            description = input("Enter script description: ")
        
        if not description:
            self.print_colored("❌ Please provide a description for script generation.", 'error')
            return
        
        self.print_colored(f"🤖 Generating script: {description}", 'info')
        
        # For script generation, we need to break down the description into commands
        # This is a simplified approach - could be enhanced with more sophisticated parsing
        commands = []
        
        # Try to generate individual commands from the description
        command_result = self.command_generator.generate_command(description)
        if command_result.get("success"):
            commands.append(command_result["command"])
        
        if not commands:
            self.print_colored("❌ Could not generate commands for the script.", 'error')
            return
        
        # Generate the script
        script_result = self.command_generator.generate_script(description, commands)
        
        if script_result.get("success"):
            script_content = script_result["script_content"]
            script_name = script_result["script_name"]
            safety = script_result["safety_assessment"]
            
            self.print_colored("✅ Generated Script:", 'success')
            self.print_colored(f"📄 Script Name: {script_name}", 'info')
            self.print_colored(f"🛡️  Safety Level: {safety['overall_safety']}", 
                              'warning' if safety['overall_safety'] != 'safe' else 'success')
            
            print("\n" + "="*50)
            print(script_content)
            print("="*50 + "\n")
            
            # Ask if user wants to save the script
            save = input("Save this script to file? (y/N): ").lower().strip()
            if save == 'y':
                save_result = self.command_generator.save_script(script_content, script_name)
                if save_result.get("success"):
                    self.print_colored(f"✅ Script saved to: {save_result['script_path']}", 'success')
                    self.print_colored(f"📏 Size: {save_result['size']} bytes", 'info')
                    if save_result.get("executable"):
                        self.print_colored("🔧 Script is executable", 'success')
                else:
                    self.print_colored(f"❌ Failed to save script: {save_result.get('error')}", 'error')
        else:
            self.print_colored(f"❌ Failed to generate script: {script_result.get('error', 'Unknown error')}", 'error')
    
    def run(self):
        """Run the CLI interface."""
        self.running = True
        self.print_banner()
        
        # Check ClamAV availability
        if not self.clamav_scanner.is_available():
            self.print_colored("⚠️  ClamAV not detected. Install with: sudo apt install clamav clamav-daemon", 'warning')
        
        self.print_colored("Type 'help' for available commands or 'exit' to quit.\n", 'info')
        
        while self.running:
            try:
                # Get user input
                command = input(f"{self.colors['bold']}linux-ai-agent> {self.colors['reset']}").strip()
                
                if not command:
                    continue
                
                # Execute command
                should_continue = self.execute_command(command)
                if not should_continue:
                    self.running = False
                
                print()  # Add spacing between commands
                
            except KeyboardInterrupt:
                print()
                self.print_colored("👋 Goodbye!", 'success')
                self.running = False
            except EOFError:
                print()
                self.print_colored("👋 Goodbye!", 'success')
                self.running = False
            except Exception as e:
                self.print_colored(f"❌ Unexpected error: {e}", 'error')
                logger.error(f"Unexpected error in CLI: {e}")


def main():
    """Main entry point for CLI."""
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
