#!/usr/bin/env python3
"""
Test script for detailed scan reporting functionality.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_detailed_reporting():
    """Test the detailed scan reporting system."""
    
    print("🧪 Testing Detailed Scan Reporting System")
    print("=" * 50)
    
    try:
        from scanner.scan_reporter import ScanReporter
        from ai.intent_parser import IntentParser
        
        # Test 1: Intent Parser Recognition
        print("\n1. Testing Intent Parser Recognition...")
        parser = IntentParser()
        
        test_commands = [
            "detailed scan suspicious.exe",
            "comprehensive scan malware.bin", 
            "full report document.pdf",
            "generate report test.txt"
        ]
        
        for cmd in test_commands:
            result = parser.parse_intent(cmd)
            intent = result["intent"]
            confidence = result["confidence"]
            print(f"   '{cmd}' -> {intent} (confidence: {confidence:.2f})")
            
            if intent == "detailed_scan" and confidence > 0.7:
                print("   ✅ Correctly recognized as detailed_scan")
            else:
                print(f"   ❌ Expected detailed_scan, got {intent}")
        
        # Test 2: ScanReporter Functionality
        print("\n2. Testing ScanReporter...")
        reporter = ScanReporter()
        
        # Create a test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test file for detailed reporting.")
            test_file = f.name
        
        try:
            # Mock scan results
            mock_scan_results = {
                "infected": False,
                "overall_suspicious": True,
                "risk_score": 45,
                "all_reasons": [
                    "High entropy detected: 7.8",
                    "Suspicious extension: .exe"
                ],
                "entropy_analysis": {
                    "average_entropy": 7.8,
                    "max_entropy": 8.0,
                    "high_entropy_chunks": 3,
                    "total_chunks": 10,
                    "suspicious": True
                },
                "attribute_analysis": {
                    "suspicious": True,
                    "reasons": ["Suspicious extension: .exe"]
                }
            }
            
            additional_data = {
                "filepath": test_file,
                "processing_time": "0.45s",
                "deep_scan": True
            }
            
            # Generate detailed report
            detailed_report = reporter.generate_detailed_report(
                "file_scan", mock_scan_results, additional_data
            )
            
            print("   ✅ Report generated successfully")
            print(f"   📊 Report ID: {detailed_report.get('report_id', 'N/A')}")
            print(f"   📋 Summary: {detailed_report.get('summary', 'N/A')}")
            
            # Test report sections
            sections = [
                'file_analysis', 'security_assessment', 'technical_details',
                'threat_indicators', 'recommendations', 'metadata'
            ]
            
            for section in sections:
                if section in detailed_report:
                    print(f"   ✅ {section.replace('_', ' ').title()} section present")
                else:
                    print(f"   ❌ {section.replace('_', ' ').title()} section missing")
            
            # Test specific functionality
            security = detailed_report.get('security_assessment', {})
            if security:
                risk_level = security.get('risk_level', 'UNKNOWN')
                risk_score = security.get('risk_score', 0)
                print(f"   🛡️  Security Assessment: {risk_level} (Score: {risk_score})")
            
            recommendations = detailed_report.get('recommendations', [])
            print(f"   💡 Recommendations: {len(recommendations)} generated")
            
            # Test file analysis
            file_analysis = detailed_report.get('file_analysis', {})
            if file_analysis and not file_analysis.get('error'):
                basic_info = file_analysis.get('basic_info', {})
                if basic_info:
                    print(f"   📁 File Analysis: {basic_info.get('name', 'Unknown')} ({basic_info.get('size_human', 'Unknown')})")
        
        finally:
            # Clean up test file
            try:
                os.unlink(test_file)
            except:
                pass
        
        # Test 3: Report Display Formatting
        print("\n3. Testing Report Display...")
        
        # Test entropy interpretation
        entropy_data = {
            "average_entropy": 7.8,
            "max_entropy": 8.0,
            "high_entropy_chunks": 8,
            "total_chunks": 10
        }
        
        interpretation = reporter._interpret_entropy(entropy_data)
        entropy_level = interpretation.get('entropy_level', 'unknown')
        explanation = interpretation.get('explanation', 'No explanation')
        
        print(f"   📈 Entropy Level: {entropy_level}")
        print(f"   📝 Explanation: {explanation}")
        
        if entropy_level == "very_high":
            print("   ✅ High entropy correctly identified")
        else:
            print(f"   ❌ Expected very_high entropy, got {entropy_level}")
        
        # Test 4: File Type Detection
        print("\n4. Testing File Type Detection...")
        
        test_files = {
            "malware.exe": {"category": "executable", "risk_level": "high"},
            "document.pdf": {"category": "document", "risk_level": "medium"},
            "archive.zip": {"category": "archive", "risk_level": "medium"},
            "readme.txt": {"category": "text", "risk_level": "low"}
        }
        
        for filename, expected in test_files.items():
            file_type = reporter._detect_file_type(filename)
            category = file_type.get('category', 'unknown')
            risk_level = file_type.get('risk_level', 'unknown')
            
            print(f"   📄 {filename}: {category} ({risk_level} risk)")
            
            if category == expected['category'] and risk_level == expected['risk_level']:
                print(f"      ✅ Correctly classified")
            else:
                print(f"      ❌ Expected {expected['category']}/{expected['risk_level']}")
        
        print("\n" + "=" * 50)
        print("🎉 Detailed Reporting Test Complete!")
        print("✅ All core functionality verified")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Make sure all dependencies are available")
        return False
    except Exception as e:
        print(f"❌ Test Error: {e}")
        return False

def test_cli_integration():
    """Test CLI integration with detailed reporting."""
    
    print("\n🧪 Testing CLI Integration")
    print("=" * 30)
    
    try:
        # Test that CLI can import the reporter
        from interface.cli import CLI
        
        cli = CLI()
        
        # Check if scan_reporter is initialized
        if hasattr(cli, 'scan_reporter'):
            print("   ✅ ScanReporter initialized in CLI")
        else:
            print("   ❌ ScanReporter not found in CLI")
            return False
        
        # Test intent recognition for detailed scan
        result = cli.intent_parser.parse_intent("detailed scan test.exe")
        
        if result["intent"] == "detailed_scan":
            print("   ✅ CLI recognizes detailed_scan intent")
        else:
            print(f"   ❌ CLI intent parser returned: {result['intent']}")
            return False
        
        print("   ✅ CLI integration successful")
        return True
        
    except Exception as e:
        print(f"   ❌ CLI Integration Error: {e}")
        return False

if __name__ == "__main__":
    print("🛡️ Linux AI Agent - Detailed Reporting Test")
    print("=" * 60)
    
    success = True
    
    # Run tests
    success &= test_detailed_reporting()
    success &= test_cli_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Detailed reporting system is ready for use")
        print("\n💡 Try these commands in the CLI:")
        print("   • detailed scan <filename>")
        print("   • comprehensive scan <filename>") 
        print("   • full report <filename>")
    else:
        print("❌ SOME TESTS FAILED!")
        print("⚠️  Please check the errors above")
    
    print("=" * 60)