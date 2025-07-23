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
from ai import IntentParser, RLAgent
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

⚙️ General:
  • help                        - Show this help
  • exit                        - Exit the application

Examples:
  scan file /home/user/download.zip
  create file test.txt
  quarantine file suspicious.exe
  disk usage
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
        if heuristic_result["overall_suspicious"]:
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
