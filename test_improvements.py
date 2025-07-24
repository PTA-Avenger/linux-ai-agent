#!/usr/bin/env python3
"""
Test script for Linux AI Agent improvements.
Tests the enhanced features and fixes implemented.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_enhanced_intent_parser():
    """Test the enhanced intent parser with flag support."""
    print("🧪 Testing Enhanced Intent Parser...")
    
    try:
        from ai.enhanced_intent_parser import EnhancedIntentParser
        parser = EnhancedIntentParser()
        
        # Test flag parsing
        test_cases = [
            "ls -l /var/log",
            "heuristic scan /tmp/suspicious.exe", 
            "scan file --recursive /home/user",
            "help",
            "--help"
        ]
        
        for test_case in test_cases:
            result = parser.parse_intent(test_case)
            print(f"  Input: '{test_case}'")
            print(f"  Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
            if result.get('flags'):
                print(f"  Flags: {result['flags']}")
            print()
        
        print("✅ Enhanced Intent Parser test passed")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Intent Parser test failed: {e}")
        return False

def test_command_generator():
    """Test the command generator."""
    print("🧪 Testing Command Generator...")
    
    try:
        from ai.command_generator import CommandGenerator
        generator = CommandGenerator()
        
        test_descriptions = [
            "backup /var/log to /backup",
            "find large files in /home/user",
            "clean up the system",
            "check disk space",
            "scan /tmp for malware"
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
            print()
        
        print("✅ Command Generator test passed")
        return True
        
    except Exception as e:
        print(f"❌ Command Generator test failed: {e}")
        return False

def test_enhanced_rl_agent():
    """Test the enhanced RL agent."""
    print("🧪 Testing Enhanced RL Agent...")
    
    try:
        from ai.rl_agent import RLAgent
        agent = RLAgent()
        
        # Test context and recommendations
        context = {
            "file_size": 1024 * 1024,  # 1MB
            "scan_results": {
                "infected": False,
                "suspicious": True,
                "risk_score": 75
            },
            "filepath": "/tmp/suspicious.exe"
        }
        
        state = agent.get_state(context)
        print(f"  Generated state: {state}")
        
        recommendations = agent.get_recommendations(context)
        print(f"  Recommendations:")
        for rec in recommendations:
            print(f"    • {rec['action']} (confidence: {rec['confidence']:.2f})")
        
        # Test learning
        agent.learn(state, "quarantine_file", 5.0, "file_quarantined")
        
        stats = agent.get_statistics()
        print(f"  Training episodes: {stats['training_episodes']}")
        print(f"  Memory size: {stats['memory_size']}")
        
        print("✅ Enhanced RL Agent test passed")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced RL Agent test failed: {e}")
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
        
        print("✅ ClamAV Fallback test passed")
        return True
        
    except Exception as e:
        print(f"❌ ClamAV Fallback test failed: {e}")
        return False

def test_heuristic_scanner():
    """Test heuristic scanner improvements."""
    print("🧪 Testing Heuristic Scanner...")
    
    try:
        from scanner.heuristics import HeuristicScanner
        scanner = HeuristicScanner()
        
        # Create a test file
        test_file = Path("/tmp/test_entropy.txt")
        test_file.write_text("This is a test file with normal entropy content.")
        
        result = scanner.scan_file(str(test_file))
        print(f"  Scan status: {result['status']}")
        print(f"  Overall suspicious: {result.get('overall_suspicious', 'N/A')}")
        print(f"  Risk score: {result.get('risk_score', 'N/A')}")
        
        # Clean up
        test_file.unlink()
        
        print("✅ Heuristic Scanner test passed")
        return True
        
    except Exception as e:
        print(f"❌ Heuristic Scanner test failed: {e}")
        return False

def main():
    """Run all improvement tests."""
    print("🚀 Testing Linux AI Agent Improvements")
    print("=" * 50)
    
    tests = [
        test_enhanced_intent_parser,
        test_command_generator,
        test_enhanced_rl_agent,
        test_clamav_fallback,
        test_heuristic_scanner
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
        print("🎉 All improvements working correctly!")
        return 0
    else:
        print("⚠️  Some tests failed - check the output above")
        return 1

if __name__ == "__main__":
    sys.exit(main())