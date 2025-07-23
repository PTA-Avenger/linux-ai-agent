#!/usr/bin/env python3
"""
Basic test script for Linux AI Agent
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        from crud import create_file, read_file, update_file, delete_file
        print("✅ CRUD modules imported successfully")
        
        from monitor import get_disk_usage
        print("✅ Monitor modules imported successfully")
        
        from scanner import ClamAVScanner, HeuristicScanner, QuarantineManager
        print("✅ Scanner modules imported successfully")
        
        from ai import IntentParser, RLAgent
        print("✅ AI modules imported successfully")
        
        from interface import CLI
        print("✅ Interface module imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test file operations
        from crud import create_file, read_file, delete_file
        
        test_file = "test_file.txt"
        test_content = "Hello, Linux AI Agent!"
        
        # Create file
        if create_file(test_file, test_content):
            print("✅ File creation works")
        else:
            print("❌ File creation failed")
            return False
        
        # Read file
        content = read_file(test_file)
        if content == test_content:
            print("✅ File reading works")
        else:
            print("❌ File reading failed")
            return False
        
        # Clean up
        if delete_file(test_file):
            print("✅ File deletion works")
        else:
            print("❌ File deletion failed")
        
        # Test intent parser
        from ai import IntentParser
        parser = IntentParser()
        result = parser.parse_intent("scan file test.txt")
        
        if result["intent"] == "scan_file":
            print("✅ Intent parsing works")
        else:
            print("❌ Intent parsing failed")
            return False
        
        # Test disk usage
        from monitor import get_disk_usage
        usage = get_disk_usage("/")
        
        if usage and "total_gb" in usage:
            print("✅ Disk usage monitoring works")
        else:
            print("❌ Disk usage monitoring failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def main():
    """Run basic tests."""
    print("🛡️  Linux AI Agent - Basic Tests")
    print("=" * 40)
    
    success = True
    
    success &= test_imports()
    success &= test_basic_functionality()
    
    print("\n" + "=" * 40)
    if success:
        print("✅ All basic tests passed!")
        print("🚀 Application is ready to use.")
        print("\nTo start the application:")
        print("  python3 src/main.py")
    else:
        print("❌ Some tests failed.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
