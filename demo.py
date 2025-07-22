#!/usr/bin/env python3
"""
Demo script showing Linux AI Agent functionality
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def main():
    """Demonstrate the Linux AI Agent functionality."""
    print("🛡️ Linux AI Agent - Demo")
    print("=" * 50)
    
    # Import modules
    from crud import create_file, read_file, delete_file, list_directory
    from monitor import get_disk_usage, get_system_stats
    from scanner import ClamAVScanner, HeuristicScanner, QuarantineManager
    from ai import IntentParser, RLAgent
    
    # 1. File Operations Demo
    print("\n�� File Operations Demo:")
    demo_file = "demo_file.txt"
    demo_content = "This is a demo file for the Linux AI Agent!"
    
    if create_file(demo_file, demo_content):
        print(f"✅ Created file: {demo_file}")
        
        content = read_file(demo_file)
        print(f"✅ Read file content: {content[:30]}...")
        
        if delete_file(demo_file):
            print(f"✅ Deleted file: {demo_file}")
    
    # 2. System Monitoring Demo
    print("\n📊 System Monitoring Demo:")
    disk_usage = get_disk_usage("/")
    print(f"✅ Disk usage: {disk_usage['used_percent']:.1f}% used, {disk_usage['free_gb']:.1f}GB free")
    
    system_stats = get_system_stats()
    print(f"✅ System stats: CPU {system_stats['cpu_percent']:.1f}%, Memory {system_stats['memory']['used_percent']:.1f}%")
    
    # 3. Scanner Demo
    print("\n🛡️ Scanner Demo:")
    clamav = ClamAVScanner()
    if clamav.is_available():
        print("✅ ClamAV is available")
    else:
        print("⚠️ ClamAV not available (install with: sudo apt install clamav)")
    
    heuristic = HeuristicScanner()
    print("✅ Heuristic scanner initialized")
    
    quarantine = QuarantineManager()
    stats = quarantine.get_quarantine_stats()
    print(f"✅ Quarantine manager: {stats['total_files']} files quarantined")
    
    # 4. AI Demo
    print("\n🤖 AI Demo:")
    parser = IntentParser()
    
    test_commands = [
        "create file test.txt with content hello world",
        "scan directory /home for malware",
        "show disk usage for /var",
        "list files in current directory"
    ]
    
    for cmd in test_commands:
        result = parser.parse_intent(cmd)
        print(f"✅ '{cmd}' -> Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
    
    rl_agent = RLAgent()
    recommendations = rl_agent.get_recommendations({"scan_results": {"suspicious_files": 0}})
    print(f"✅ RL Agent recommendations: {len(recommendations)} suggestions")
    
    print("\n" + "=" * 50)
    print("🚀 Demo completed successfully!")
    print("\nTo use the interactive CLI:")
    print("  python3 src/main.py")

if __name__ == "__main__":
    main()
