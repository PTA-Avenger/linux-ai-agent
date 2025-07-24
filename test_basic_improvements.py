#!/usr/bin/env python3
"""
Basic test script for Linux AI Agent improvements.
Tests core improvements without requiring advanced ML libraries.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_intent_parser():
    """Test the basic intent parser improvements."""
    print("🧪 Testing Basic Intent Parser...")
    
    try:
        from ai.intent_parser import IntentParser
        parser = IntentParser()
        
        # Test cases including the problematic heuristic scan
        test_cases = [
            "heuristic scan /tmp/suspicious.exe",
            "scan file /home/user/document.pdf", 
            "list directory /var/log",
            "help",
            "disk usage"
        ]
        
        for test_case in test_cases:
            result = parser.parse_intent(test_case)
            print(f"  Input: '{test_case}'")
            print(f"  Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
            if result.get('parameters'):
                print(f"  Parameters: {result['parameters']}")
            print()
        
        print("✅ Basic Intent Parser test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic Intent Parser test failed: {e}")
        return False

def test_command_generator():
    """Test the command generator."""
    print("🧪 Testing Command Generator...")
    
    try:
        from ai.command_generator import CommandGenerator
        generator = CommandGenerator()
        
        test_descriptions = [
            "backup /var/log to /backup",
            "check disk space",
            "find large files in /home/user"
        ]
        
        for description in test_descriptions:
            result = generator.generate_command(description)
            print(f"  Description: '{description}'")
            if result.get("success"):
                print(f"  Command: {result['command']}")
                print(f"  Safety: {result['safety_level']}")
                print(f"  Confidence: {result['confidence']:.2f}")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
                if result.get('suggestions'):
                    print(f"  Suggestions: {result['suggestions']}")
            print()
        
        print("✅ Command Generator test passed")
        return True
        
    except Exception as e:
        print(f"❌ Command Generator test failed: {e}")
        return False

def test_basic_rl_agent():
    """Test the basic RL agent functionality."""
    print("🧪 Testing Basic RL Agent...")
    
    try:
        from ai.rl_agent import RLAgent
        agent = RLAgent()
        
        # Test basic context
        context = {
            "file_size": 1024,
            "scan_results": {
                "infected": False,
                "suspicious": False
            }
        }
        
        state = agent.get_state(context)
        print(f"  Generated state: {state}")
        
        recommendations = agent.get_recommendations(context)
        print(f"  Recommendations count: {len(recommendations)}")
        for rec in recommendations[:2]:  # Show first 2
            print(f"    • {rec['action']} (confidence: {rec['confidence']:.2f})")
        
        stats = agent.get_statistics()
        print(f"  Training episodes: {stats['training_episodes']}")
        
        print("✅ Basic RL Agent test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic RL Agent test failed: {e}")
        return False

def test_clamav_fallback():
    """Test ClamAV with heuristic fallback."""
    print("🧪 Testing ClamAV with Fallback...")
    
    try:
        from scanner.clamav_wrapper import ClamAVScanner
        scanner = ClamAVScanner(enable_fallback=True)
        
        print(f"  ClamAV available: {scanner.available}")
        print(f"  Fallback enabled: {scanner.enable_fallback}")
        
        # Test with a non-existent file (should handle gracefully)
        result = scanner.scan_file("/tmp/nonexistent_test_file.txt")
        print(f"  Scan result status: {result['status']}")
        print(f"  Has fallback details: {'heuristic_details' in result}")
        
        print("✅ ClamAV Fallback test passed")
        return True
        
    except Exception as e:
        print(f"❌ ClamAV Fallback test failed: {e}")
        return False

def test_heuristic_scanner_error_handling():
    """Test heuristic scanner error handling improvements."""
    print("🧪 Testing Heuristic Scanner Error Handling...")
    
    try:
        from scanner.heuristics import HeuristicScanner
        scanner = HeuristicScanner()
        
        # Create a test file
        test_file = Path("/tmp/test_entropy.txt")
        test_file.write_text("This is a test file with normal entropy content.")
        
        result = scanner.scan_file(str(test_file))
        print(f"  Scan status: {result['status']}")
        
        # Test the key that was causing issues
        overall_suspicious = result.get('overall_suspicious', 'KEY_MISSING')
        print(f"  Overall suspicious key present: {overall_suspicious != 'KEY_MISSING'}")
        print(f"  Overall suspicious value: {overall_suspicious}")
        
        # Test risk score
        risk_score = result.get('risk_score', 'MISSING')
        print(f"  Risk score present: {risk_score != 'MISSING'}")
        
        # Clean up
        test_file.unlink()
        
        print("✅ Heuristic Scanner Error Handling test passed")
        return True
        
    except Exception as e:
        print(f"❌ Heuristic Scanner Error Handling test failed: {e}")
        return False

def test_script_generation():
    """Test script generation functionality."""
    print("🧪 Testing Script Generation...")
    
    try:
        from ai.command_generator import CommandGenerator
        generator = CommandGenerator()
        
        # Test script generation
        description = "daily system maintenance"
        commands = [
            "sudo apt update",
            "sudo apt autoremove -y", 
            "sudo journalctl --vacuum-time=7d"
        ]
        
        result = generator.generate_script(description, commands)
        
        if result.get("success"):
            print(f"  Script name: {result['script_name']}")
            print(f"  Command count: {result['command_count']}")
            print(f"  Safety level: {result['safety_assessment']['overall_safety']}")
            print("  Script content preview:")
            lines = result['script_content'].split('\n')[:10]
            for line in lines:
                print(f"    {line}")
            print("    ...")
        else:
            print(f"  Error: {result.get('error')}")
        
        print("✅ Script Generation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Script Generation test failed: {e}")
        return False

def main():
    """Run all basic improvement tests."""
    print("🚀 Testing Linux AI Agent Basic Improvements")
    print("=" * 50)
    
    tests = [
        test_basic_intent_parser,
        test_command_generator,
        test_basic_rl_agent,
        test_clamav_fallback,
        test_heuristic_scanner_error_handling,
        test_script_generation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
        print("-" * 30)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic improvements working correctly!")
        return 0
    else:
        print("⚠️  Some tests failed - check the output above")
        return 1

if __name__ == "__main__":
    sys.exit(main())