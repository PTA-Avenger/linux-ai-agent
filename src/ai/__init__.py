"""
AI components package for Linux AI Agent.
"""

from .intent_parser import IntentParser
from .rl_agent import RLAgent

# Enhanced AI components (optional imports)
try:
    from .enhanced_intent_parser import EnhancedIntentParser
    ENHANCED_INTENT_PARSER_AVAILABLE = True
except ImportError:
    ENHANCED_INTENT_PARSER_AVAILABLE = False

try:
    from .advanced_rl_agent import AdvancedRLAgent
    ADVANCED_RL_AGENT_AVAILABLE = True
except ImportError:
    ADVANCED_RL_AGENT_AVAILABLE = False

try:
    from .ml_malware_detector import MLMalwareDetector
    ML_MALWARE_DETECTOR_AVAILABLE = True
except ImportError:
    ML_MALWARE_DETECTOR_AVAILABLE = False

# Base exports (always available)
__all__ = ['IntentParser', 'RLAgent']

# Add enhanced components if available
if ENHANCED_INTENT_PARSER_AVAILABLE:
    __all__.append('EnhancedIntentParser')

if ADVANCED_RL_AGENT_AVAILABLE:
    __all__.append('AdvancedRLAgent')

if ML_MALWARE_DETECTOR_AVAILABLE:
    __all__.append('MLMalwareDetector')
