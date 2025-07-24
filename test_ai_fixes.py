#!/usr/bin/env python3
"""
Test script to verify AI command parsing fixes.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ai.intent_parser import IntentParser

def test_ai_command_parsing():
    """Test that AI-related commands are properly parsed."""
    parser = IntentParser()
    
    test_cases = [
        ("ai stats", "ai_stats"),
        ("show ai stats", "ai_stats"),
        ("ai recommend file security", "ai_recommend"),
        ("what do you recommend for system monitoring", "ai_recommend"),
        ("generate command to backup files", "generate_command"),
        ("create script for log cleanup", "generate_script"),
        ("heuristic scan suspicious.exe", "heuristic_scan"),
    ]
    
    print("🧪 Testing AI Command Parsing...")
    print("=" * 50)
    
    all_passed = True
    
    for command, expected_intent in test_cases:
        result = parser.parse_intent(command)
        actual_intent = result["intent"]
        confidence = result["confidence"]
        
        if actual_intent == expected_intent and confidence > 0.5:
            print(f"✅ '{command}' -> {actual_intent} (confidence: {confidence:.2f})")
        else:
            print(f"❌ '{command}' -> {actual_intent} (confidence: {confidence:.2f}) [Expected: {expected_intent}]")
            all_passed = False
    
    print("=" * 50)
    return all_passed

def test_fuzzy_suggestions():
    """Test fuzzy command suggestions for malformed input."""
    parser = IntentParser()
    
    test_cases = [
        "crcreate file",
        "ai stat",
        "generat command",
        "scann file",
        "delet quarantine"
    ]
    
    print("\n🔍 Testing Fuzzy Command Suggestions...")
    print("=" * 50)
    
    for malformed_command in test_cases:
        suggestions = parser.suggest_commands(malformed_command, limit=3)
        print(f"'{malformed_command}' -> Suggestions: {suggestions}")
    
    print("=" * 50)

def test_parameter_extraction():
    """Test parameter extraction for different command types."""
    parser = IntentParser()
    
    test_cases = [
        ("ai recommend system security", {"context": "system security"}),
        ("generate command to list files", {"description": "to list files"}),
        ("scan file test.exe", {"path": "test.exe"}),
    ]
    
    print("\n📋 Testing Parameter Extraction...")
    print("=" * 50)
    
    all_passed = True
    
    for command, expected_params in test_cases:
        result = parser.parse_intent(command)
        parameters = result["parameters"]
        
        # Check if expected parameters are present
        match = True
        for key, expected_value in expected_params.items():
            if key not in parameters or expected_value not in parameters[key]:
                match = False
                break
        
        if match:
            print(f"✅ '{command}' -> {parameters}")
        else:
            print(f"❌ '{command}' -> {parameters} [Expected to contain: {expected_params}]")
            all_passed = False
    
    print("=" * 50)
    return all_passed

if __name__ == "__main__":
    print("🚀 Running AI Command Parsing Tests\n")
    
    test1_passed = test_ai_command_parsing()
    test_fuzzy_suggestions()
    test2_passed = test_parameter_extraction()
    
    print(f"\n📊 Test Results:")
    print(f"AI Command Parsing: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Parameter Extraction: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All critical tests passed! AI command parsing is working correctly.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
        sys.exit(1)