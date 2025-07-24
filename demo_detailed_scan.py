#!/usr/bin/env python3
"""
Demo script showcasing the enhanced detailed scan reporting functionality.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_demo_files():
    """Create demo files for testing."""
    demo_files = {}
    
    # Create a suspicious executable-like file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.exe', delete=False) as f:
        # Write high-entropy data to simulate packed/encrypted content
        import random
        random.seed(42)  # For reproducible results
        high_entropy_data = bytes([random.randint(0, 255) for _ in range(8192)])
        f.write(high_entropy_data)
        demo_files['suspicious_exe'] = f.name
    
    # Create a normal text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a normal text file with regular content.\n" * 100)
        demo_files['normal_txt'] = f.name
    
    # Create a document file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
        f.write("PDF-like content with some structure and text.\n" * 50)
        demo_files['document_pdf'] = f.name
    
    return demo_files

def demo_detailed_reporting():
    """Demonstrate the detailed reporting system."""
    
    print("🛡️ Linux AI Agent - Enhanced Detailed Scan Reporting Demo")
    print("=" * 70)
    
    try:
        from scanner.scan_reporter import ScanReporter
        from scanner.heuristics import HeuristicScanner
        from ai.intent_parser import IntentParser
        
        # Initialize components
        reporter = ScanReporter()
        scanner = HeuristicScanner()
        parser = IntentParser()
        
        print("\n🎯 Demonstrating Enhanced Scan Reporting Features:")
        print("   ✅ Comprehensive file analysis with hashes")
        print("   ✅ Advanced entropy interpretation")
        print("   ✅ Security risk assessment with scoring")
        print("   ✅ Detailed threat indicators")
        print("   ✅ Actionable recommendations")
        print("   ✅ Compliance checking")
        print("   ✅ Professional report formatting")
        
        # Create demo files
        print("\n📁 Creating demo files for testing...")
        demo_files = create_demo_files()
        
        for file_type, filepath in demo_files.items():
            print(f"   Created: {file_type} -> {Path(filepath).name}")
        
        # Demo 1: Intent Recognition
        print("\n" + "=" * 70)
        print("🧠 DEMO 1: Enhanced Intent Recognition")
        print("=" * 70)
        
        test_commands = [
            "detailed scan suspicious.exe",
            "comprehensive scan document.pdf", 
            "full report malware.bin",
            "generate report archive.zip"
        ]
        
        for cmd in test_commands:
            result = parser.parse_intent(cmd)
            intent = result["intent"]
            confidence = result["confidence"]
            parameters = result.get("parameters", {})
            
            print(f"\n🔍 Command: '{cmd}'")
            print(f"   📊 Intent: {intent}")
            print(f"   🎯 Confidence: {confidence:.2f}")
            if parameters:
                print(f"   📋 Parameters: {parameters}")
        
        # Demo 2: Detailed Analysis of Suspicious File
        print("\n" + "=" * 70)
        print("🚨 DEMO 2: Suspicious File Analysis")
        print("=" * 70)
        
        suspicious_file = demo_files['suspicious_exe']
        print(f"\n🔍 Analyzing suspicious file: {Path(suspicious_file).name}")
        
        # Perform heuristic scan
        heuristic_result = scanner.scan_file(suspicious_file)
        
        # Create comprehensive scan results
        scan_results = {
            "infected": False,  # No known virus signature
            "overall_suspicious": heuristic_result.get("overall_suspicious", False),
            "risk_score": heuristic_result.get("risk_score", 0),
            "all_reasons": heuristic_result.get("all_reasons", []),
            "entropy_analysis": heuristic_result.get("entropy_analysis", {}),
            "attribute_analysis": heuristic_result.get("attribute_analysis", {}),
            "scan_type": "detailed"
        }
        
        additional_data = {
            "filepath": suspicious_file,
            "processing_time": "1.23s",
            "deep_scan": True
        }
        
        # Generate detailed report
        detailed_report = reporter.generate_detailed_report(
            "file_scan", scan_results, additional_data
        )
        
        # Display key sections
        print(f"\n📋 REPORT SUMMARY:")
        print(f"   🆔 Report ID: {detailed_report.get('report_id')}")
        print(f"   📊 Summary: {detailed_report.get('summary')}")
        
        # Security Assessment
        security = detailed_report.get('security_assessment', {})
        if security:
            risk_level = security.get('risk_level')
            risk_score = security.get('risk_score')
            print(f"\n🛡️  SECURITY ASSESSMENT:")
            print(f"   📊 Risk Level: {risk_level}")
            print(f"   🔢 Risk Score: {risk_score}/100")
            
            risk_factors = security.get('risk_factors', [])
            if risk_factors:
                print(f"   ⚠️  Risk Factors:")
                for factor in risk_factors[:3]:
                    print(f"      • {factor}")
        
        # File Analysis
        file_analysis = detailed_report.get('file_analysis', {})
        if file_analysis and not file_analysis.get('error'):
            print(f"\n📁 FILE ANALYSIS:")
            
            basic_info = file_analysis.get('basic_info', {})
            if basic_info:
                print(f"   📏 Size: {basic_info.get('size_human')}")
                print(f"   📝 Type: {basic_info.get('extension')}")
            
            file_type = file_analysis.get('file_type', {})
            if file_type:
                category = file_type.get('category')
                risk_level = file_type.get('risk_level')
                print(f"   🏷️  Category: {category} ({risk_level} risk)")
            
            identity = file_analysis.get('identity', {})
            if identity.get('sha256'):
                print(f"   🔐 SHA256: {identity['sha256'][:32]}...")
        
        # Threat Indicators
        indicators = detailed_report.get('threat_indicators', [])
        if indicators:
            print(f"\n🚩 THREAT INDICATORS:")
            for i, indicator in enumerate(indicators[:3], 1):
                severity = indicator.get('severity')
                confidence = indicator.get('confidence', 0)
                description = indicator.get('description')
                print(f"   {i}. {description}")
                print(f"      Severity: {severity.upper()}, Confidence: {confidence:.1%}")
        
        # Recommendations
        recommendations = detailed_report.get('recommendations', [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations[:2], 1):
                priority = rec.get('priority')
                description = rec.get('description')
                print(f"   {i}. [{priority}] {description}")
                
                steps = rec.get('steps', [])
                if steps:
                    print(f"      Steps:")
                    for step in steps[:2]:
                        print(f"        • {step}")
        
        # Demo 3: Clean File Analysis
        print("\n" + "=" * 70)
        print("✅ DEMO 3: Clean File Analysis")
        print("=" * 70)
        
        clean_file = demo_files['normal_txt']
        print(f"\n🔍 Analyzing clean file: {Path(clean_file).name}")
        
        # Perform heuristic scan on clean file
        clean_heuristic = scanner.scan_file(clean_file)
        
        clean_results = {
            "infected": False,
            "overall_suspicious": clean_heuristic.get("overall_suspicious", False),
            "risk_score": clean_heuristic.get("risk_score", 0),
            "all_reasons": clean_heuristic.get("all_reasons", []),
            "entropy_analysis": clean_heuristic.get("entropy_analysis", {}),
            "scan_type": "detailed"
        }
        
        clean_additional = {
            "filepath": clean_file,
            "processing_time": "0.15s",
            "deep_scan": True
        }
        
        clean_report = reporter.generate_detailed_report(
            "file_scan", clean_results, clean_additional
        )
        
        print(f"\n📋 CLEAN FILE REPORT:")
        print(f"   📊 Summary: {clean_report.get('summary')}")
        
        clean_security = clean_report.get('security_assessment', {})
        if clean_security:
            print(f"   🛡️  Risk Level: {clean_security.get('risk_level')}")
            print(f"   📊 Risk Score: {clean_security.get('risk_score')}/100")
        
        clean_compliance = clean_report.get('compliance_check', {})
        if clean_compliance:
            compliance_level = clean_compliance.get('compliance_level')
            compliance_score = clean_compliance.get('compliance_score')
            print(f"   ✅ Compliance: {compliance_level} ({compliance_score}/100)")
        
        # Demo 4: Advanced Features
        print("\n" + "=" * 70)
        print("🔬 DEMO 4: Advanced Analysis Features")
        print("=" * 70)
        
        print("\n📈 Entropy Interpretation:")
        entropy_data = {
            "average_entropy": 7.8,
            "max_entropy": 8.0,
            "high_entropy_chunks": 8,
            "total_chunks": 10
        }
        
        interpretation = reporter._interpret_entropy(entropy_data)
        print(f"   Level: {interpretation.get('entropy_level')}")
        print(f"   Explanation: {interpretation.get('explanation')}")
        
        concerns = interpretation.get('concerns', [])
        if concerns:
            print(f"   Concerns:")
            for concern in concerns:
                print(f"     • {concern}")
        
        print("\n🏷️  File Type Classification:")
        test_files = ['malware.exe', 'document.pdf', 'archive.zip', 'script.py']
        for filename in test_files:
            file_type = reporter._detect_file_type(filename)
            category = file_type.get('category')
            risk_level = file_type.get('risk_level')
            description = file_type.get('description')
            print(f"   {filename}: {category} ({risk_level} risk) - {description}")
        
        print("\n" + "=" * 70)
        print("🎉 DEMO COMPLETE!")
        print("=" * 70)
        
        print("\n✅ Enhanced Features Demonstrated:")
        print("   • Professional detailed reporting")
        print("   • Comprehensive security assessment")
        print("   • Advanced entropy analysis and interpretation")
        print("   • File type classification with risk levels")
        print("   • Actionable recommendations with steps")
        print("   • Compliance checking")
        print("   • File integrity verification (hashes)")
        print("   • Threat indicator categorization")
        
        print("\n💡 Available Commands in CLI:")
        print("   • detailed scan <filename>     - Generate comprehensive report")
        print("   • comprehensive scan <filename> - Same as detailed scan")
        print("   • full report <filename>       - Generate detailed analysis")
        print("   • scan file <filename>         - Quick scan with basic reporting")
        print("   • heuristic scan <filename>    - Heuristic analysis only")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up demo files
        try:
            for filepath in demo_files.values():
                os.unlink(filepath)
        except:
            pass

if __name__ == "__main__":
    success = demo_detailed_reporting()
    
    if success:
        print("\n🎊 Demo completed successfully!")
        print("The enhanced detailed scan reporting system is ready for use!")
    else:
        print("\n❌ Demo encountered errors.")