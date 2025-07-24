"""
Advanced scan reporting module for Linux AI Agent.
Provides detailed, formatted reports for various scan types.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from utils import get_logger, log_operation

logger = get_logger("scan_reporter")


class ScanReporter:
    """Advanced scan report generator with detailed analysis and formatting."""
    
    def __init__(self):
        self.report_templates = {
            "file_scan": self._generate_file_scan_report,
            "directory_scan": self._generate_directory_scan_report,
            "heuristic_scan": self._generate_heuristic_report,
            "clamav_scan": self._generate_clamav_report
        }
    
    def generate_detailed_report(self, scan_type: str, scan_results: Dict[str, Any], 
                               additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive detailed report for any scan type.
        
        Args:
            scan_type: Type of scan performed
            scan_results: Raw scan results
            additional_data: Additional context data
            
        Returns:
            Formatted detailed report
        """
        try:
            report_generator = self.report_templates.get(scan_type, self._generate_generic_report)
            
            base_report = {
                "report_id": self._generate_report_id(),
                "timestamp": datetime.now().isoformat(),
                "scan_type": scan_type,
                "summary": self._generate_summary(scan_results),
                "raw_results": scan_results
            }
            
            detailed_report = report_generator(scan_results, additional_data or {})
            base_report.update(detailed_report)
            
            # Add recommendations
            base_report["recommendations"] = self._generate_recommendations(scan_results, scan_type)
            
            # Add metadata
            base_report["metadata"] = self._generate_metadata(scan_results, additional_data)
            
            return base_report
            
        except Exception as e:
            logger.error(f"Error generating detailed report: {e}")
            return {
                "error": f"Report generation failed: {str(e)}",
                "raw_results": scan_results
            }
    
    def _generate_file_scan_report(self, scan_results: Dict[str, Any], 
                                  additional_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed file scan report."""
        filepath = additional_data.get("filepath") or scan_results.get("filepath", "Unknown")
        
        report = {
            "file_analysis": self._analyze_file_properties(filepath),
            "security_assessment": self._assess_security_risk(scan_results),
            "technical_details": self._extract_technical_details(scan_results),
            "threat_indicators": self._identify_threat_indicators(scan_results),
            "compliance_check": self._check_compliance(filepath, scan_results)
        }
        
        return report
    
    def _check_compliance(self, filepath: str, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check file compliance with security policies."""
        compliance_issues = []
        compliance_score = 100
        
        # Check for infected files
        if scan_results.get("infected", False):
            compliance_issues.append("File contains known malware")
            compliance_score -= 100
        
        # Check for suspicious files
        if scan_results.get("overall_suspicious", False):
            risk_score = scan_results.get("risk_score", 0)
            compliance_issues.append(f"File flagged as suspicious (Risk: {risk_score}%)")
            compliance_score -= risk_score * 0.5
        
        # Check file type restrictions
        file_path = Path(filepath)
        extension = file_path.suffix.lower()
        
        restricted_extensions = ['.exe', '.bat', '.cmd', '.scr', '.vbs', '.js']
        if extension in restricted_extensions:
            compliance_issues.append(f"Potentially dangerous file type: {extension}")
            compliance_score -= 20
        
        compliance_level = "COMPLIANT"
        if compliance_score < 50:
            compliance_level = "NON_COMPLIANT"
        elif compliance_score < 80:
            compliance_level = "PARTIAL_COMPLIANCE"
        
        return {
            "compliance_score": max(int(compliance_score), 0),
            "compliance_level": compliance_level,
            "issues": compliance_issues,
            "recommendations": self._get_compliance_recommendations(compliance_issues)
        }
    
    def _get_compliance_recommendations(self, issues: List[str]) -> List[str]:
        """Get compliance recommendations based on issues."""
        recommendations = []
        
        for issue in issues:
            if "malware" in issue.lower():
                recommendations.append("Immediately quarantine and remove infected file")
            elif "suspicious" in issue.lower():
                recommendations.append("Perform additional security analysis")
            elif "dangerous file type" in issue.lower():
                recommendations.append("Verify file source and necessity before use")
        
        if not recommendations:
            recommendations.append("File meets current security compliance standards")
        
        return recommendations
    
    def _generate_heuristic_report(self, scan_results: Dict[str, Any], 
                                  additional_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed heuristic analysis report."""
        entropy_analysis = scan_results.get("entropy_analysis", {})
        attribute_analysis = scan_results.get("attribute_analysis", {})
        
        report = {
            "entropy_breakdown": {
                "analysis": entropy_analysis,
                "interpretation": self._interpret_entropy(entropy_analysis),
                "risk_factors": self._identify_entropy_risks(entropy_analysis)
            },
            "attribute_breakdown": {
                "analysis": attribute_analysis,
                "interpretation": self._interpret_attributes(attribute_analysis),
                "risk_factors": self._identify_attribute_risks(attribute_analysis)
            },
            "behavioral_indicators": self._analyze_behavioral_indicators(scan_results),
            "pattern_analysis": self._analyze_patterns(scan_results),
            "confidence_metrics": self._calculate_confidence_metrics(scan_results)
        }
        
        return report
    
    def _generate_clamav_report(self, scan_results: Dict[str, Any], 
                               additional_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed ClamAV scan report."""
        report = {
            "virus_database_info": self._get_virus_db_info(),
            "scan_performance": self._analyze_scan_performance(additional_data),
            "threat_classification": self._classify_threat(scan_results),
            "signature_analysis": self._analyze_signatures(scan_results),
            "false_positive_assessment": self._assess_false_positive_risk(scan_results)
        }
        
        return report
    
    def _generate_directory_scan_report(self, scan_results: Dict[str, Any], 
                                       additional_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed directory scan report."""
        report = {
            "directory_structure": self._analyze_directory_structure(scan_results),
            "file_type_distribution": self._analyze_file_types(scan_results),
            "risk_distribution": self._analyze_risk_distribution(scan_results),
            "hotspots": self._identify_security_hotspots(scan_results),
            "coverage_analysis": self._analyze_scan_coverage(scan_results)
        }
        
        return report
    
    def _analyze_file_properties(self, filepath: str) -> Dict[str, Any]:
        """Analyze comprehensive file properties."""
        try:
            file_path = Path(filepath)
            
            if not file_path.exists():
                return {"error": "File does not exist"}
            
            stat = file_path.stat()
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(filepath)
            
            properties = {
                "basic_info": {
                    "name": file_path.name,
                    "extension": file_path.suffix,
                    "size_bytes": stat.st_size,
                    "size_human": self._format_file_size(stat.st_size),
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "accessed": datetime.fromtimestamp(stat.st_atime).isoformat()
                },
                "permissions": {
                    "mode": oct(stat.st_mode),
                    "readable": os.access(filepath, os.R_OK),
                    "writable": os.access(filepath, os.W_OK),
                    "executable": os.access(filepath, os.X_OK)
                },
                "identity": {
                    "md5": file_hash.get("md5"),
                    "sha1": file_hash.get("sha1"),
                    "sha256": file_hash.get("sha256")
                },
                "file_type": self._detect_file_type(filepath)
            }
            
            return properties
            
        except Exception as e:
            return {"error": f"Failed to analyze file properties: {str(e)}"}
    
    def _assess_security_risk(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess comprehensive security risk."""
        risk_factors = []
        risk_score = 0
        risk_level = "LOW"
        
        # ClamAV results
        if scan_results.get("infected", False):
            risk_factors.append("Known malware signature detected")
            risk_score += 90
            risk_level = "CRITICAL"
        
        # Heuristic results
        if scan_results.get("overall_suspicious", False):
            heuristic_risk = scan_results.get("risk_score", 0)
            risk_factors.append(f"Heuristic analysis flagged (Risk: {heuristic_risk}%)")
            risk_score += heuristic_risk * 0.7
        
        # Determine final risk level
        if risk_score >= 80:
            risk_level = "CRITICAL"
        elif risk_score >= 60:
            risk_level = "HIGH"
        elif risk_score >= 30:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "risk_score": min(int(risk_score), 100),
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "mitigation_priority": self._get_mitigation_priority(risk_level),
            "threat_categories": self._categorize_threats(scan_results)
        }
    
    def _extract_technical_details(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive technical details."""
        details = {
            "scan_methods": [],
            "detection_engines": [],
            "analysis_depth": "standard",
            "processing_time": scan_results.get("processing_time", "unknown")
        }
        
        # Add ClamAV details
        if "clamav" in str(scan_results.get("scan_type", "")):
            details["scan_methods"].append("signature_based")
            details["detection_engines"].append("ClamAV")
            
            if scan_results.get("virus_name"):
                details["signature_match"] = scan_results["virus_name"]
        
        # Add heuristic details
        if "heuristic" in str(scan_results.get("scan_type", "")):
            details["scan_methods"].append("heuristic_analysis")
            details["detection_engines"].append("Custom Heuristic Engine")
            details["analysis_depth"] = "deep"
            
            entropy_data = scan_results.get("entropy_analysis", {})
            if entropy_data:
                details["entropy_metrics"] = {
                    "average": entropy_data.get("average_entropy", 0),
                    "maximum": entropy_data.get("max_entropy", 0),
                    "chunks_analyzed": entropy_data.get("total_chunks", 0),
                    "high_entropy_chunks": entropy_data.get("high_entropy_chunks", 0)
                }
        
        return details
    
    def _identify_threat_indicators(self, scan_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific threat indicators."""
        indicators = []
        
        # Process all reasons from scan results
        all_reasons = scan_results.get("all_reasons", [])
        for reason in all_reasons:
            indicator = {
                "type": self._classify_indicator_type(reason),
                "description": reason,
                "severity": self._assess_indicator_severity(reason),
                "confidence": self._assess_indicator_confidence(reason)
            }
            indicators.append(indicator)
        
        # Add entropy-based indicators
        entropy_analysis = scan_results.get("entropy_analysis", {})
        if entropy_analysis.get("suspicious", False):
            indicators.append({
                "type": "entropy_anomaly",
                "description": f"High entropy detected (avg: {entropy_analysis.get('average_entropy', 0):.2f})",
                "severity": "medium",
                "confidence": 0.8
            })
        
        return indicators
    
    def _interpret_entropy(self, entropy_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Provide detailed entropy interpretation."""
        avg_entropy = entropy_analysis.get("average_entropy", 0)
        max_entropy = entropy_analysis.get("max_entropy", 0)
        high_chunks = entropy_analysis.get("high_entropy_chunks", 0)
        total_chunks = entropy_analysis.get("total_chunks", 1)
        
        interpretation = {
            "entropy_level": "normal",
            "explanation": "File appears to have normal entropy levels",
            "concerns": [],
            "technical_notes": []
        }
        
        if avg_entropy > 7.5:
            interpretation["entropy_level"] = "very_high"
            interpretation["explanation"] = "File shows very high entropy, indicating possible encryption, compression, or obfuscation"
            interpretation["concerns"].append("Potential malware obfuscation")
            interpretation["concerns"].append("Encrypted or compressed content")
        elif avg_entropy > 6.5:
            interpretation["entropy_level"] = "high"
            interpretation["explanation"] = "File shows elevated entropy levels"
            interpretation["concerns"].append("Possible compressed or encoded content")
        elif avg_entropy > 5.0:
            interpretation["entropy_level"] = "moderate"
            interpretation["explanation"] = "File shows moderate entropy levels"
        
        if high_chunks > total_chunks * 0.5:
            interpretation["concerns"].append(f"Many high-entropy sections ({high_chunks}/{total_chunks})")
        
        interpretation["technical_notes"].append(f"Average entropy: {avg_entropy:.3f} bits per byte")
        interpretation["technical_notes"].append(f"Maximum entropy: {max_entropy:.3f} bits per byte")
        interpretation["technical_notes"].append(f"Entropy threshold: 7.5 bits per byte")
        
        return interpretation
    
    def _generate_recommendations(self, scan_results: Dict[str, Any], scan_type: str) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on scan results."""
        recommendations = []
        
        # Critical threat recommendations
        if scan_results.get("infected", False):
            recommendations.append({
                "priority": "CRITICAL",
                "action": "immediate_quarantine",
                "description": "Immediately quarantine the infected file",
                "rationale": "Known malware signature detected",
                "steps": [
                    "Stop any running processes using this file",
                    "Move file to secure quarantine location",
                    "Run full system scan",
                    "Check for lateral movement"
                ]
            })
        
        # Suspicious file recommendations
        elif scan_results.get("overall_suspicious", False):
            risk_score = scan_results.get("risk_score", 0)
            
            if risk_score > 70:
                recommendations.append({
                    "priority": "HIGH",
                    "action": "detailed_analysis",
                    "description": "Perform detailed malware analysis",
                    "rationale": f"High risk score ({risk_score}%) detected",
                    "steps": [
                        "Isolate file for analysis",
                        "Run additional scanning tools",
                        "Analyze file behavior in sandbox",
                        "Consider expert review"
                    ]
                })
            elif risk_score > 30:
                recommendations.append({
                    "priority": "MEDIUM",
                    "action": "enhanced_monitoring",
                    "description": "Monitor file and related processes",
                    "rationale": f"Moderate risk score ({risk_score}%) detected",
                    "steps": [
                        "Monitor file access patterns",
                        "Check for unusual network activity",
                        "Verify file legitimacy",
                        "Consider backup before use"
                    ]
                })
        
        # General security recommendations
        recommendations.append({
            "priority": "LOW",
            "action": "regular_scanning",
            "description": "Maintain regular security scanning schedule",
            "rationale": "Proactive security maintenance",
            "steps": [
                "Schedule daily quick scans",
                "Update virus definitions regularly",
                "Monitor system for unusual activity",
                "Keep security tools updated"
            ]
        })
        
        return recommendations
    
    def _generate_metadata(self, scan_results: Dict[str, Any], additional_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive metadata."""
        return {
            "scan_engine_versions": {
                "heuristic_engine": "1.0.0",
                "clamav_version": "unknown",  # Could be enhanced to get actual version
                "signature_database": "unknown"
            },
            "scan_parameters": {
                "deep_scan": additional_data.get("deep_scan", False),
                "entropy_threshold": 7.5,
                "chunk_size": 4096
            },
            "system_context": {
                "timestamp": datetime.now().isoformat(),
                "hostname": os.uname().nodename if hasattr(os, 'uname') else "unknown",
                "scan_user": os.getenv("USER", "unknown")
            }
        }
    
    # Helper methods
    def _generate_report_id(self) -> str:
        """Generate unique report ID."""
        timestamp = str(int(time.time()))
        return f"SCAN_{timestamp}_{hash(timestamp) % 10000:04d}"
    
    def _generate_summary(self, scan_results: Dict[str, Any]) -> str:
        """Generate executive summary."""
        if scan_results.get("infected", False):
            virus_name = scan_results.get("virus_name", "Unknown")
            return f"INFECTED: Malware detected - {virus_name}"
        elif scan_results.get("overall_suspicious", False):
            risk_score = scan_results.get("risk_score", 0)
            return f"SUSPICIOUS: Potential threat detected (Risk: {risk_score}%)"
        else:
            return "CLEAN: No threats detected"
    
    def _calculate_file_hash(self, filepath: str) -> Dict[str, str]:
        """Calculate multiple hash values for file."""
        hashes = {"md5": "", "sha1": "", "sha256": ""}
        
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
                hashes["md5"] = hashlib.md5(content).hexdigest()
                hashes["sha1"] = hashlib.sha1(content).hexdigest()
                hashes["sha256"] = hashlib.sha256(content).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hashes for {filepath}: {e}")
        
        return hashes
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def _detect_file_type(self, filepath: str) -> Dict[str, Any]:
        """Detect file type and characteristics."""
        file_path = Path(filepath)
        extension = file_path.suffix.lower()
        
        type_info = {
            "extension": extension,
            "category": "unknown",
            "risk_level": "low",
            "description": "Unknown file type"
        }
        
        # Categorize by extension
        if extension in ['.exe', '.dll', '.sys', '.bat', '.cmd', '.com', '.scr', '.pif']:
            type_info.update({
                "category": "executable",
                "risk_level": "high",
                "description": "Executable file - requires careful handling"
            })
        elif extension in ['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.pdf']:
            type_info.update({
                "category": "document",
                "risk_level": "medium",
                "description": "Document file - may contain macros"
            })
        elif extension in ['.zip', '.rar', '.7z', '.tar', '.gz']:
            type_info.update({
                "category": "archive",
                "risk_level": "medium",
                "description": "Archive file - contents unknown"
            })
        elif extension in ['.txt', '.log', '.cfg', '.ini']:
            type_info.update({
                "category": "text",
                "risk_level": "low",
                "description": "Text file - generally safe"
            })
        
        return type_info
    
    def _classify_indicator_type(self, reason: str) -> str:
        """Classify threat indicator type."""
        reason_lower = reason.lower()
        
        if "entropy" in reason_lower:
            return "entropy_anomaly"
        elif "extension" in reason_lower:
            return "suspicious_extension"
        elif "name" in reason_lower:
            return "suspicious_filename"
        elif "size" in reason_lower:
            return "file_size_anomaly"
        else:
            return "general_suspicious"
    
    def _assess_indicator_severity(self, reason: str) -> str:
        """Assess severity of threat indicator."""
        reason_lower = reason.lower()
        
        if any(word in reason_lower for word in ['high', 'critical', 'dangerous']):
            return "high"
        elif any(word in reason_lower for word in ['medium', 'moderate', 'elevated']):
            return "medium"
        else:
            return "low"
    
    def _assess_indicator_confidence(self, reason: str) -> float:
        """Assess confidence level of threat indicator."""
        # Simple confidence assessment based on indicator type
        reason_lower = reason.lower()
        
        if "entropy" in reason_lower:
            return 0.8  # High confidence in entropy analysis
        elif "extension" in reason_lower:
            return 0.9  # Very high confidence in extension matching
        elif "signature" in reason_lower:
            return 0.95  # Highest confidence in signature detection
        else:
            return 0.6  # Moderate confidence for other indicators
    
    def _get_mitigation_priority(self, risk_level: str) -> str:
        """Get mitigation priority based on risk level."""
        priority_map = {
            "CRITICAL": "Immediate action required",
            "HIGH": "Action required within 24 hours",
            "MEDIUM": "Action required within 1 week",
            "LOW": "Monitor and review regularly"
        }
        return priority_map.get(risk_level, "Review as needed")
    
    def _categorize_threats(self, scan_results: Dict[str, Any]) -> List[str]:
        """Categorize detected threats."""
        categories = []
        
        if scan_results.get("infected", False):
            categories.append("known_malware")
        
        if scan_results.get("overall_suspicious", False):
            reasons = scan_results.get("all_reasons", [])
            for reason in reasons:
                if "entropy" in reason.lower():
                    categories.append("obfuscation")
                elif "extension" in reason.lower():
                    categories.append("suspicious_filetype")
        
        return categories if categories else ["no_threats"]
    
    def _generate_generic_report(self, scan_results: Dict[str, Any], 
                               additional_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate generic report for unknown scan types."""
        return {
            "generic_analysis": {
                "status": scan_results.get("status", "unknown"),
                "results_summary": str(scan_results),
                "note": "Generic report - specific analysis not available for this scan type"
            }
        }
    
    # Placeholder methods for future enhancement
    def _get_virus_db_info(self) -> Dict[str, Any]:
        return {"version": "unknown", "last_update": "unknown"}
    
    def _analyze_scan_performance(self, additional_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"processing_time": additional_data.get("processing_time", "unknown")}
    
    def _classify_threat(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        return {"classification": "unknown"}
    
    def _analyze_signatures(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        return {"signature_match": scan_results.get("virus_name", "none")}
    
    def _assess_false_positive_risk(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        return {"risk": "unknown"}
    
    def _analyze_directory_structure(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        return {"structure": "unknown"}
    
    def _analyze_file_types(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        return {"distribution": "unknown"}
    
    def _analyze_risk_distribution(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        return {"distribution": "unknown"}
    
    def _identify_security_hotspots(self, scan_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []
    
    def _analyze_scan_coverage(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        return {"coverage": "unknown"}
    
    def _identify_entropy_risks(self, entropy_analysis: Dict[str, Any]) -> List[str]:
        risks = []
        if entropy_analysis.get("average_entropy", 0) > 7.5:
            risks.append("Very high entropy - possible obfuscation")
        if entropy_analysis.get("high_entropy_chunks", 0) > 5:
            risks.append("Multiple high-entropy sections detected")
        return risks
    
    def _interpret_attributes(self, attribute_analysis: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "suspicious": attribute_analysis.get("suspicious", False),
            "explanation": "File attributes analyzed for suspicious characteristics"
        }
    
    def _identify_attribute_risks(self, attribute_analysis: Dict[str, Any]) -> List[str]:
        return attribute_analysis.get("reasons", [])
    
    def _analyze_behavioral_indicators(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        return {"indicators": "none detected"}
    
    def _analyze_patterns(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        return {"patterns": "none detected"}
    
    def _calculate_confidence_metrics(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "overall_confidence": 0.8,
            "method": "heuristic_analysis"
        }