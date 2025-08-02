"""
Command Line Interface for Linux AI Agent.
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from crud import create_file, read_file, update_file, delete_file
from monitor import get_disk_usage, monitor_disk_space, get_file_activity
from scanner import ClamAVScanner, HeuristicScanner, QuarantineManager
from scanner.scan_reporter import ScanReporter
from ai import IntentParser, RLAgent, CommandGenerator
try:
    from ai import GemmaAgent
    GEMMA_AVAILABLE = True
except ImportError:
    GemmaAgent = None
    GEMMA_AVAILABLE = False
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
        self.scan_reporter = ScanReporter()
        
        # Initialize Gemma agent if available
        self.gemma_agent = None
        if GEMMA_AVAILABLE:
            try:
                self.gemma_agent = GemmaAgent(mode="auto")
                logger.info("Gemma agent initialized successfully")
            except Exception as e:
                logger.warning(f"Gemma agent initialization failed: {e}")
                self.gemma_agent = None
        
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
  • detailed scan <path>        - Generate comprehensive security report

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
  • clean up                    - System cleanup recommendations
  • gemma <question>            - Chat with Gemma AI assistant
  • ai analyze <topic>          - AI-powered analysis of topics or files

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
            
            # Try to suggest similar commands
            suggestions = self.intent_parser.suggest_commands(command, limit=3)
            if suggestions:
                self.print_colored("💡 Did you mean:", 'info')
                for suggestion in suggestions:
                    self.print_colored(f"    • {suggestion}", 'info')
            else:
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
            
            elif intent == "detailed_scan":
                self._handle_detailed_scan(parameters)
            
            elif intent == "generate_command":
                self._handle_generate_command(parameters, command)
            
            elif intent == "generate_script":
                self._handle_generate_script(parameters, command)
            
            elif intent == "ai_stats":
                self._handle_ai_stats()
            
            elif intent == "ai_recommend":
                self._handle_ai_recommend(parameters, command)
            
            elif intent == "system_cleanup":
                self._handle_system_cleanup(parameters)
            
            elif intent == "gemma_chat":
                self._handle_gemma_chat(parameters, command)
            
            elif intent == "ai_analyze":
                self._handle_ai_analyze(parameters, command)
            
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
        
        # Offer detailed report option for suspicious files
        if heuristic_result.get("overall_suspicious", False) or clamav_result.get("infected", False):
            self.print_colored("\n💡 Tip: Use 'detailed scan <filename>' for comprehensive analysis report", 'info')
    
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
        
        # Try Gemma first for enhanced command generation
        if self.gemma_agent:
            try:
                self.print_colored("🔮 Using Gemma AI for command generation...", 'info')
                gemma_result = self.gemma_agent.generate_system_command(description)
                
                if gemma_result.get("success"):
                    command = gemma_result["command"]
                    explanation = gemma_result["explanation"]
                    safety_level = gemma_result["safety_level"]
                    confidence = gemma_result["confidence"]
                    model_used = gemma_result["model_used"]
                    
                    self.print_colored("✅ Gemma Generated Command:", 'success')
                    self.print_colored(f"📝 Command: {command}", 'info')
                    self.print_colored(f"📖 Explanation: {explanation}", 'info')
                    self.print_colored(f"🎯 Confidence: {confidence:.2f}", 'info')
                    self.print_colored(f"🛡️  Safety Level: {safety_level}", 
                                      'warning' if safety_level != 'safe' else 'success')
                    self.print_colored(f"🔮 Model: {model_used}", 'info')
                    
                    # Ask if user wants to execute the command
                    if safety_level != 'dangerous' and command:
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
                    elif safety_level == 'dangerous':
                        self.print_colored("⚠️  Command marked as dangerous - execution blocked for safety", 'error')
                    
                    return  # Use Gemma result and skip basic generator
                else:
                    logger.warning(f"Gemma command generation failed: {gemma_result.get('error')}")
                    self.print_colored("⚠️  Gemma unavailable, using basic generator...", 'warning')
                    
            except Exception as e:
                logger.warning(f"Gemma command generation error: {e}")
                self.print_colored("⚠️  Gemma error, using basic generator...", 'warning')
        
        # Fallback to basic command generator
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

    def _handle_detailed_scan(self, parameters: Dict[str, Any]):
        """Handle detailed file scanning with comprehensive reporting."""
        if "path" not in parameters:
            self.print_colored("❌ Please specify a file path for detailed scan.", 'error')
            return
        
        filepath = parameters["path"]
        self.print_colored(f"🔍 Performing detailed scan on '{filepath}'...", 'info')
        self.print_colored("📊 Generating comprehensive report...", 'info')
        
        start_time = time.time()
        
        # Perform all scans
        clamav_result = self.clamav_scanner.scan_file(filepath)
        heuristic_result = self.heuristic_scanner.scan_file(filepath)
        
        processing_time = time.time() - start_time
        
        # Combine results for detailed analysis
        combined_results = {
            "infected": clamav_result.get("infected", False),
            "virus_name": clamav_result.get("virus_name"),
            "overall_suspicious": heuristic_result.get("overall_suspicious", False),
            "risk_score": heuristic_result.get("risk_score", 0),
            "all_reasons": heuristic_result.get("all_reasons", []),
            "entropy_analysis": heuristic_result.get("entropy_analysis", {}),
            "attribute_analysis": heuristic_result.get("attribute_analysis", {}),
            "processing_time": f"{processing_time:.2f}s",
            "scan_type": "detailed"
        }
        
        # Generate detailed report
        additional_data = {
            "filepath": filepath,
            "processing_time": f"{processing_time:.2f}s",
            "deep_scan": True
        }
        
        detailed_report = self.scan_reporter.generate_detailed_report(
            "file_scan", combined_results, additional_data
        )
        
        # Display comprehensive report
        self._display_detailed_report(detailed_report, filepath)
    
    def _display_detailed_report(self, report: Dict[str, Any], filepath: str):
        """Display a comprehensive detailed scan report."""
        self.print_colored("\n" + "=" * 80, 'header')
        self.print_colored("📋 COMPREHENSIVE SCAN REPORT", 'header')
        self.print_colored("=" * 80, 'header')
        
        # Report header
        self.print_colored(f"📁 File: {Path(filepath).name}", 'info')
        self.print_colored(f"🆔 Report ID: {report.get('report_id', 'N/A')}", 'info')
        self.print_colored(f"⏰ Timestamp: {report.get('timestamp', 'N/A')}", 'info')
        
        # Executive summary
        summary = report.get('summary', 'No summary available')
        if 'INFECTED' in summary:
            self.print_colored(f"🚨 EXECUTIVE SUMMARY: {summary}", 'error')
        elif 'SUSPICIOUS' in summary:
            self.print_colored(f"⚠️  EXECUTIVE SUMMARY: {summary}", 'warning')
        else:
            self.print_colored(f"✅ EXECUTIVE SUMMARY: {summary}", 'success')
        
        # File analysis
        file_analysis = report.get('file_analysis', {})
        if file_analysis and not file_analysis.get('error'):
            self.print_colored("\n📊 FILE ANALYSIS:", 'header')
            
            basic_info = file_analysis.get('basic_info', {})
            if basic_info:
                self.print_colored(f"    📏 Size: {basic_info.get('size_human', 'Unknown')}", 'info')
                self.print_colored(f"    📝 Type: {basic_info.get('extension', 'Unknown')}", 'info')
                self.print_colored(f"    📅 Modified: {basic_info.get('modified', 'Unknown')[:19]}", 'info')
            
            file_type = file_analysis.get('file_type', {})
            if file_type:
                category = file_type.get('category', 'unknown')
                risk_level = file_type.get('risk_level', 'unknown')
                description = file_type.get('description', 'No description')
                
                risk_color = 'error' if risk_level == 'high' else 'warning' if risk_level == 'medium' else 'success'
                self.print_colored(f"    🏷️  Category: {category.title()} ({risk_level.upper()} risk)", risk_color)
                self.print_colored(f"    📄 Description: {description}", 'info')
            
            # File hashes
            identity = file_analysis.get('identity', {})
            if identity and identity.get('sha256'):
                self.print_colored(f"    🔐 SHA256: {identity['sha256'][:32]}...", 'info')
        
        # Security assessment
        security = report.get('security_assessment', {})
        if security:
            self.print_colored("\n🛡️  SECURITY ASSESSMENT:", 'header')
            
            risk_level = security.get('risk_level', 'UNKNOWN')
            risk_score = security.get('risk_score', 0)
            
            risk_color = 'error' if risk_level == 'CRITICAL' else 'warning' if risk_level in ['HIGH', 'MEDIUM'] else 'success'
            self.print_colored(f"    📊 Risk Level: {risk_level} (Score: {risk_score}/100)", risk_color)
            self.print_colored(f"    ⏰ Priority: {security.get('mitigation_priority', 'Unknown')}", 'info')
            
            risk_factors = security.get('risk_factors', [])
            if risk_factors:
                self.print_colored("    ⚠️  Risk Factors:", 'warning')
                for factor in risk_factors:
                    self.print_colored(f"        • {factor}", 'warning')
            
            threat_categories = security.get('threat_categories', [])
            if threat_categories and threat_categories != ['no_threats']:
                self.print_colored(f"    🎯 Threat Categories: {', '.join(threat_categories)}", 'warning')
        
        # Threat indicators
        indicators = report.get('threat_indicators', [])
        if indicators:
            self.print_colored("\n🚩 THREAT INDICATORS:", 'header')
            for i, indicator in enumerate(indicators[:10], 1):
                severity = indicator.get('severity', 'unknown')
                confidence = indicator.get('confidence', 0)
                description = indicator.get('description', 'No description')
                
                severity_color = 'error' if severity == 'high' else 'warning' if severity == 'medium' else 'info'
                self.print_colored(f"    {i}. {description}", severity_color)
                self.print_colored(f"       Severity: {severity.upper()}, Confidence: {confidence:.1%}", 'info')
            
            if len(indicators) > 10:
                self.print_colored(f"    ... and {len(indicators) - 10} more indicators", 'info')
        
        # Technical details
        technical = report.get('technical_details', {})
        if technical:
            self.print_colored("\n🔧 TECHNICAL DETAILS:", 'header')
            
            scan_methods = technical.get('scan_methods', [])
            if scan_methods:
                self.print_colored(f"    🔍 Scan Methods: {', '.join(scan_methods)}", 'info')
            
            detection_engines = technical.get('detection_engines', [])
            if detection_engines:
                self.print_colored(f"    🤖 Detection Engines: {', '.join(detection_engines)}", 'info')
            
            processing_time = technical.get('processing_time', 'unknown')
            self.print_colored(f"    ⏱️  Processing Time: {processing_time}", 'info')
            
            entropy_metrics = technical.get('entropy_metrics', {})
            if entropy_metrics:
                self.print_colored("    📈 Entropy Analysis:", 'info')
                self.print_colored(f"        Average: {entropy_metrics.get('average', 0):.3f} bits/byte", 'info')
                self.print_colored(f"        Maximum: {entropy_metrics.get('maximum', 0):.3f} bits/byte", 'info')
                self.print_colored(f"        Chunks: {entropy_metrics.get('chunks_analyzed', 0)} analyzed", 'info')
                self.print_colored(f"        High Entropy: {entropy_metrics.get('high_entropy_chunks', 0)} chunks", 'info')
        
        # Compliance check
        compliance = report.get('compliance_check', {})
        if compliance:
            self.print_colored("\n📋 COMPLIANCE CHECK:", 'header')
            
            compliance_level = compliance.get('compliance_level', 'UNKNOWN')
            compliance_score = compliance.get('compliance_score', 0)
            
            compliance_color = 'success' if compliance_level == 'COMPLIANT' else 'warning' if compliance_level == 'PARTIAL_COMPLIANCE' else 'error'
            self.print_colored(f"    ✅ Status: {compliance_level} (Score: {compliance_score}/100)", compliance_color)
            
            issues = compliance.get('issues', [])
            if issues:
                self.print_colored("    ⚠️  Issues:", 'warning')
                for issue in issues:
                    self.print_colored(f"        • {issue}", 'warning')
        
        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            self.print_colored("\n💡 RECOMMENDATIONS:", 'header')
            
            for i, rec in enumerate(recommendations[:5], 1):
                priority = rec.get('priority', 'UNKNOWN')
                action = rec.get('action', 'No action specified')
                description = rec.get('description', 'No description')
                rationale = rec.get('rationale', 'No rationale provided')
                
                priority_color = 'error' if priority == 'CRITICAL' else 'warning' if priority == 'HIGH' else 'info'
                self.print_colored(f"    {i}. [{priority}] {description}", priority_color)
                self.print_colored(f"       Action: {action}", 'info')
                self.print_colored(f"       Rationale: {rationale}", 'info')
                
                steps = rec.get('steps', [])
                if steps:
                    self.print_colored("       Steps:", 'info')
                    for step in steps[:3]:
                        self.print_colored(f"         • {step}", 'info')
                    if len(steps) > 3:
                        self.print_colored(f"         ... and {len(steps) - 3} more steps", 'info')
                
                if i < len(recommendations):
                    self.print_colored("", 'info')  # Empty line between recommendations
        
        # Report footer
        self.print_colored("\n" + "=" * 80, 'header')
        self.print_colored("📄 End of Report", 'header')
        self.print_colored("=" * 80 + "\n", 'header')
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    def _handle_system_cleanup(self, parameters: Dict[str, Any]):
        """Handle system cleanup operations."""
        self.print_colored("🧹 System Cleanup Operations", 'header')
        
        # Generate cleanup script
        cleanup_description = "clean up system files, logs, and temporary data"
        
        self.print_colored("🤖 Generating system cleanup script...", 'info')
        
        result = self.command_generator.generate_command(cleanup_description)
        
        if result.get("success"):
            command = result["command"]
            confidence = result["confidence"]
            safety_level = result["safety_level"]
            
            self.print_colored("✅ Generated Cleanup Commands:", 'success')
            self.print_colored(f"📝 Command: {command}", 'info')
            self.print_colored(f"🎯 Confidence: {confidence:.2f}", 'info')
            self.print_colored(f"🛡️  Safety Level: {safety_level}", 
                              'warning' if safety_level != 'safe' else 'success')
            
            # Also provide manual cleanup suggestions
            self.print_colored("\n💡 Manual Cleanup Recommendations:", 'header')
            cleanup_tasks = [
                "Clear package cache: sudo apt autoremove && sudo apt autoclean",
                "Clean log files: sudo journalctl --vacuum-time=7d",
                "Remove temporary files: sudo rm -rf /tmp/* (use with caution)",
                "Clear user cache: rm -rf ~/.cache/*",
                "Clean thumbnails: rm -rf ~/.thumbnails/*"
            ]
            
            for i, task in enumerate(cleanup_tasks, 1):
                self.print_colored(f"    {i}. {task}", 'info')
            
            self.print_colored("\n⚠️  Safety Notes:", 'warning')
            self.print_colored("    • Review commands before execution", 'warning')
            self.print_colored("    • Backup important data first", 'warning')
            self.print_colored("    • Test in non-production environment", 'warning')
            
        else:
            # Fallback to manual recommendations
            self.print_colored("💡 System Cleanup Recommendations:", 'header')
            
            cleanup_categories = {
                "🗂️  Package Management": [
                    "sudo apt update && sudo apt upgrade",
                    "sudo apt autoremove",
                    "sudo apt autoclean"
                ],
                "📝 Log Files": [
                    "sudo journalctl --vacuum-time=7d",
                    "sudo find /var/log -name '*.log' -mtime +30 -delete",
                    "sudo logrotate /etc/logrotate.conf"
                ],
                "🗃️  Temporary Files": [
                    "sudo rm -rf /tmp/*",
                    "rm -rf ~/.cache/*",
                    "rm -rf ~/.thumbnails/*"
                ],
                "💾 Disk Space": [
                    "df -h  # Check disk usage",
                    "du -sh /* | sort -hr | head -10  # Find large directories",
                    "find . -type f -size +100M  # Find large files"
                ]
            }
            
            for category, commands in cleanup_categories.items():
                self.print_colored(f"\n{category}:", 'info')
                for cmd in commands:
                    self.print_colored(f"    • {cmd}", 'info')

    def _handle_gemma_chat(self, parameters: Dict[str, Any], original_command: str):
        """Handle direct chat with Gemma AI."""
        if not self.gemma_agent:
            self.print_colored("❌ Gemma AI is not available", 'error')
            self.print_colored("💡 To enable Gemma:", 'info')
            self.print_colored("    • Install dependencies: pip install google-generativeai transformers torch", 'info')
            self.print_colored("    • Set API key: export GEMMA_API_KEY=your_key", 'info')
            self.print_colored("    • Or use local model with transformers", 'info')
            return
        
        # Extract question from parameters or original command
        question = parameters.get("path") or parameters.get("context") or ""
        
        if not question:
            # Try to extract from original command
            question = original_command.replace("gemma", "").replace("ask gemma", "").replace("chat with gemma", "").strip()
        
        if not question:
            question = input("Ask Gemma: ")
        
        if not question:
            self.print_colored("❌ Please provide a question for Gemma.", 'error')
            return
        
        self.print_colored(f"🔮 Asking Gemma: {question}", 'info')
        
        try:
            result = self.gemma_agent.generate_response(
                prompt=question,
                system_prompt="You are a helpful AI assistant for system administration and cybersecurity. Provide clear, accurate, and practical responses.",
                max_tokens=512,
                temperature=0.7
            )
            
            if result["success"]:
                response = result["response"]
                processing_time = result["processing_time"]
                mode = result["mode"]
                
                self.print_colored("🤖 Gemma Response:", 'success')
                self.print_colored(response, 'info')
                self.print_colored(f"\n📊 Response generated in {processing_time} using {mode} mode", 'info')
            else:
                error = result.get("error", "Unknown error")
                self.print_colored(f"❌ Gemma failed to respond: {error}", 'error')
                
        except Exception as e:
            logger.error(f"Gemma chat error: {e}")
            self.print_colored(f"❌ Error communicating with Gemma: {e}", 'error')

    def _handle_ai_analyze(self, parameters: Dict[str, Any], original_command: str):
        """Handle AI analysis requests."""
        if not self.gemma_agent:
            self.print_colored("❌ Gemma AI is not available for analysis", 'error')
            self.print_colored("💡 Using basic analysis instead...", 'info')
            # Fallback to basic analysis
            self._handle_ai_recommend(parameters, original_command)
            return
        
        # Extract topic from parameters or original command
        topic = parameters.get("path") or parameters.get("context") or ""
        
        if not topic:
            # Try to extract from original command
            topic = original_command.replace("ai analyze", "").replace("analyze with ai", "").strip()
        
        if not topic:
            topic = input("What would you like to analyze: ")
        
        if not topic:
            self.print_colored("❌ Please provide a topic for analysis.", 'error')
            return
        
        self.print_colored(f"🔍 Analyzing: {topic}", 'info')
        
        # Check if it's a file path for security analysis
        if os.path.exists(topic):
            self._analyze_file_with_gemma(topic)
        else:
            self._analyze_topic_with_gemma(topic)

    def _analyze_file_with_gemma(self, filepath: str):
        """Analyze a file using Gemma AI."""
        self.print_colored(f"📁 Performing AI-enhanced file analysis on: {filepath}", 'info')
        
        try:
            # First, perform regular scans
            self.print_colored("🔍 Running security scans...", 'info')
            clamav_result = self.clamav_scanner.scan_file(filepath)
            heuristic_result = self.heuristic_scanner.scan_file(filepath)
            
            # Combine scan results
            scan_results = {
                "clamav": clamav_result,
                "heuristic": heuristic_result,
                "file_path": filepath,
                "file_size": os.path.getsize(filepath) if os.path.exists(filepath) else 0
            }
            
            # Get Gemma analysis
            self.print_colored("🔮 Consulting Gemma AI for advanced analysis...", 'info')
            gemma_analysis = self.gemma_agent.analyze_security_context(scan_results, filepath)
            
            if gemma_analysis.get("analysis"):
                self.print_colored("🤖 Gemma AI Security Analysis:", 'header')
                self.print_colored(gemma_analysis["analysis"], 'info')
                
                ai_risk_score = gemma_analysis.get("ai_risk_score", 0)
                confidence = gemma_analysis.get("confidence", 0)
                model_used = gemma_analysis.get("model_used", "Unknown")
                
                self.print_colored(f"\n📊 AI Risk Score: {ai_risk_score}/100", 'warning' if ai_risk_score > 50 else 'success')
                self.print_colored(f"🎯 Analysis Confidence: {confidence:.1%}", 'info')
                self.print_colored(f"🔮 Model Used: {model_used}", 'info')
                
                # Show AI recommendations
                ai_recommendations = gemma_analysis.get("recommendations", [])
                if ai_recommendations:
                    self.print_colored("\n💡 AI Recommendations:", 'header')
                    for i, rec in enumerate(ai_recommendations, 1):
                        self.print_colored(f"    {i}. {rec}", 'info')
            else:
                error = gemma_analysis.get("error", "Unknown error")
                self.print_colored(f"❌ Gemma analysis failed: {error}", 'error')
                
        except Exception as e:
            logger.error(f"File analysis error: {e}")
            self.print_colored(f"❌ Error during file analysis: {e}", 'error')

    def _analyze_topic_with_gemma(self, topic: str):
        """Analyze a general topic using Gemma AI."""
        self.print_colored(f"🧠 Performing AI analysis on topic: {topic}", 'info')
        
        try:
            system_prompt = """You are an expert system administrator and cybersecurity analyst. 
            Provide comprehensive analysis of the given topic, focusing on:
            1. Security implications and risks
            2. Best practices and recommendations
            3. Common issues and solutions
            4. Relevant tools and techniques
            Keep your analysis practical and actionable."""
            
            result = self.gemma_agent.generate_response(
                prompt=f"Analyze this topic from a system administration and security perspective: {topic}",
                system_prompt=system_prompt,
                max_tokens=768,
                temperature=0.5
            )
            
            if result["success"]:
                response = result["response"]
                processing_time = result["processing_time"]
                mode = result["mode"]
                
                self.print_colored("🤖 Gemma AI Analysis:", 'header')
                self.print_colored(response, 'info')
                self.print_colored(f"\n📊 Analysis completed in {processing_time} using {mode} mode", 'info')
                
                # Offer follow-up actions
                self.print_colored("\n💡 Follow-up options:", 'header')
                self.print_colored("    • Use 'generate command <description>' to create related commands", 'info')
                self.print_colored("    • Use 'ai recommend <context>' for specific recommendations", 'info')
                self.print_colored("    • Use 'gemma <question>' to ask follow-up questions", 'info')
            else:
                error = result.get("error", "Unknown error")
                self.print_colored(f"❌ Gemma analysis failed: {error}", 'error')
                
        except Exception as e:
            logger.error(f"Topic analysis error: {e}")
            self.print_colored(f"❌ Error during topic analysis: {e}", 'error')

    def _handle_ai_stats(self):
        """Handle AI agent statistics display."""
        self.print_colored("🤖 AI Agent Statistics:", 'header')
        
        # RL Agent stats
        rl_stats = self.rl_agent.get_statistics()
        self.print_colored("📊 Reinforcement Learning Agent:", 'info')
        for key, value in rl_stats.items():
            self.print_colored(f"    • {key.replace('_', ' ').title()}: {value}", 'info')
        
        # Intent Parser stats
        self.print_colored("🧠 Intent Parser:", 'info')
        self.print_colored(f"    • Available Intents: {len(self.intent_parser.intent_patterns)}", 'info')
        self.print_colored(f"    • Command History: {len(self.command_history)}", 'info')
        
        # Command Generator stats  
        self.print_colored("⚙️  Command Generator:", 'info')
        self.print_colored(f"    • Available Templates: {len(self.command_generator.command_templates)}", 'info')
        
        # Gemma Agent stats
        if self.gemma_agent:
            self.print_colored("🔮 Gemma AI Agent:", 'info')
            gemma_stats = self.gemma_agent.get_statistics()
            for key, value in gemma_stats.items():
                if key == 'success_rate':
                    self.print_colored(f"    • {key.replace('_', ' ').title()}: {value:.2%}", 'info')
                else:
                    self.print_colored(f"    • {key.replace('_', ' ').title()}: {value}", 'info')
            
            # Health check
            health = self.gemma_agent.health_check()
            health_status = "✅ Healthy" if health["healthy"] else f"❌ Unhealthy ({health.get('error', 'Unknown error')})"
            self.print_colored(f"    • Health Status: {health_status}", 'success' if health["healthy"] else 'error')
        else:
            self.print_colored("🔮 Gemma AI Agent:", 'warning')
            self.print_colored("    • Status: Not Available", 'warning')
            self.print_colored("    • Reason: Missing dependencies or configuration", 'warning')
        
        print()

    def _handle_ai_recommend(self, parameters: Dict[str, Any], original_command: str):
        """Handle AI recommendations."""
        # Extract context from parameters or original command
        context_text = parameters.get("context") or parameters.get("path") or ""
        
        if not context_text:
            # Try to extract from original command
            context_text = original_command.replace("ai recommend", "").replace("recommend", "").strip()
        
        if not context_text:
            context_text = input("Enter context for recommendations: ")
        
        if not context_text:
            self.print_colored("❌ Please provide context for recommendations.", 'error')
            return
        
        self.print_colored(f"🤖 Getting AI recommendations for: {context_text}", 'info')
        
        # Try Gemma first for enhanced recommendations
        if self.gemma_agent:
            try:
                self.print_colored("🔮 Consulting Gemma AI...", 'info')
                gemma_recommendations = self.gemma_agent.provide_recommendations(context_text)
                
                if gemma_recommendations:
                    self.print_colored("💡 Gemma AI Recommendations:", 'success')
                    for i, rec in enumerate(gemma_recommendations, 1):
                        priority = rec.get('priority', 'MEDIUM')
                        description = rec.get('description', 'No description')
                        rationale = rec.get('rationale', 'No rationale')
                        
                        priority_color = 'error' if priority == 'HIGH' else 'warning' if priority == 'MEDIUM' else 'info'
                        self.print_colored(f"    {i}. [{priority}] {description}", priority_color)
                        if rationale != "AI-generated recommendation":
                            self.print_colored(f"       Rationale: {rationale}", 'info')
                    
                    return  # Use Gemma recommendations and skip RL agent
                    
            except Exception as e:
                logger.warning(f"Gemma recommendations failed: {e}")
                self.print_colored("⚠️  Gemma AI unavailable, using backup recommendations...", 'warning')
        
        # Fallback to RL agent recommendations
        context = {
            "user_input": context_text,
            "command_history": self.command_history[-5:],  # Last 5 commands
            "current_directory": os.getcwd(),
            "timestamp": time.time()
        }
        
        try:
            recommendations = self.rl_agent.get_recommendations(context)
            
            if recommendations:
                self.print_colored("💡 RL Agent Recommendations:", 'success')
                for i, rec in enumerate(recommendations, 1):
                    confidence = int(rec.get("confidence", 0) * 100)
                    action = rec.get("action", "Unknown action")
                    self.print_colored(f"    {i}. {action} (confidence: {confidence}%)", 'info')
            else:
                self.print_colored("🤔 No specific recommendations available for this context.", 'warning')
                self.print_colored("💡 General suggestions:", 'info')
                self.print_colored("    • Use 'scan file <path>' to check files for malware", 'info')
                self.print_colored("    • Use 'system stats' to monitor system health", 'info')
                self.print_colored("    • Use 'generate command <description>' for custom commands", 'info')
                
        except Exception as e:
            self.print_colored(f"❌ Error getting recommendations: {e}", 'error')
            logger.error(f"Error getting AI recommendations: {e}")


def main():
    """Main entry point for CLI."""
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
