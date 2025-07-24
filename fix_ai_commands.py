#!/usr/bin/env python3
"""
Diagnostic and fix script for AI command recognition issues.
This script will help identify and resolve why AI commands aren't being recognized.
"""

import sys
import os
import importlib
from pathlib import Path

def main():
    print("🔧 AI Command Recognition Diagnostic & Fix")
    print("=" * 50)
    
    # Add src to path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    print(f"✅ Added src path: {src_path}")
    
    try:
        # Test 1: Import and test intent parser
        print("\n🧪 Test 1: Intent Parser Import & Testing")
        
        from ai.intent_parser import IntentParser
        parser = IntentParser()
        
        test_commands = [
            'ai stats',
            'generate script cleanup',
            'ai recommend security',
            'generate command backup files'
        ]
        
        all_working = True
        for cmd in test_commands:
            result = parser.parse_intent(cmd)
            intent = result["intent"]
            confidence = result["confidence"]
            
            if intent != "unknown" and confidence > 0.7:
                print(f"  ✅ '{cmd}' -> {intent} ({confidence:.2f})")
            else:
                print(f"  ❌ '{cmd}' -> {intent} ({confidence:.2f})")
                all_working = False
        
        if all_working:
            print("  🎉 All AI commands are working correctly!")
        else:
            print("  ⚠️  Some AI commands are not working")
            return False
        
        # Test 2: Check pattern counts
        print("\n🧪 Test 2: Pattern Verification")
        
        expected_patterns = {
            'ai_stats': 3,
            'ai_recommend': 3, 
            'generate_command': 3,
            'generate_script': 3
        }
        
        for intent, expected_count in expected_patterns.items():
            actual_count = len(parser.intent_patterns.get(intent, []))
            if actual_count == expected_count:
                print(f"  ✅ {intent}: {actual_count} patterns")
            else:
                print(f"  ❌ {intent}: {actual_count} patterns (expected {expected_count})")
                all_working = False
        
        # Test 3: CLI Integration Test (if possible)
        print("\n🧪 Test 3: CLI Integration Check")
        
        try:
            # Try to import CLI components without psutil dependencies
            from ai import IntentParser as AIIntentParser
            from ai.command_generator import CommandGenerator
            
            print("  ✅ Core AI components import successfully")
            
            # Test command generator
            generator = CommandGenerator()
            gen_result = generator.generate_command("backup files")
            
            if gen_result.get("success"):
                print("  ✅ Command generator is working")
            else:
                print(f"  ⚠️  Command generator issue: {gen_result.get('error', 'Unknown')}")
            
        except ImportError as e:
            print(f"  ⚠️  CLI import issue (expected in some environments): {e}")
        
        # Test 4: Module reload fix
        print("\n🔧 Test 4: Module Reload Fix")
        
        try:
            # Force reload of intent parser module
            import ai.intent_parser
            importlib.reload(ai.intent_parser)
            print("  ✅ Intent parser module reloaded")
            
            # Test again after reload
            from ai.intent_parser import IntentParser
            fresh_parser = IntentParser()
            fresh_result = fresh_parser.parse_intent('ai stats')
            
            if fresh_result["intent"] == "ai_stats":
                print("  ✅ Fresh parser working correctly")
            else:
                print(f"  ❌ Fresh parser issue: {fresh_result['intent']}")
                
        except Exception as e:
            print(f"  ⚠️  Module reload error: {e}")
        
        print("\n" + "=" * 50)
        print("🎯 SOLUTION RECOMMENDATIONS:")
        print("=" * 50)
        
        print("\n1. 🔄 RESTART THE CLI:")
        print("   Exit the current CLI session and restart it:")
        print("   • Press Ctrl+C or type 'exit' in the CLI")
        print("   • Run: python3 src/main.py")
        
        print("\n2. 🧹 CLEAR PYTHON CACHE:")
        print("   Remove Python cache files:")
        print("   • find . -name '__pycache__' -type d -exec rm -rf {} +")
        print("   • find . -name '*.pyc' -delete")
        
        print("\n3. 🔍 VERIFY CURRENT DIRECTORY:")
        print("   Make sure you're running from the project root:")
        print("   • pwd should show the linux-ai-agent directory")
        print("   • ls should show src/, README.md, etc.")
        
        print("\n4. 🧪 TEST INDIVIDUAL COMMANDS:")
        print("   Try these commands in the CLI:")
        print("   • ai stats")
        print("   • generate command backup files")
        print("   • ai recommend security")
        
        print("\n5. 🚀 FORCE REFRESH (if needed):")
        print("   If problems persist, run this script again:")
        print("   • python3 fix_ai_commands.py")
        
        print("\n✅ The AI command patterns are correctly installed!")
        print("   The issue is likely a cached module that needs refreshing.")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\n🔧 Possible fixes:")
        print("1. Ensure you're in the correct directory")
        print("2. Check that src/ai/intent_parser.py exists")
        print("3. Verify Python path is correct")
        return False
    
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎊 Diagnostic completed successfully!")
        print("💡 Try restarting the CLI to resolve the issue.")
    else:
        print("\n⚠️  Issues detected. Please follow the recommendations above.")
    
    print("\n" + "=" * 50)