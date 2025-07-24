"""
AI module for Linux AI Agent.
"""

from .intent_parser import IntentParser
from .rl_agent import RLAgent
from .command_generator import CommandGenerator

# Optional imports with fallbacks
try:
    from .enhanced_intent_parser import EnhancedIntentParser
    ENHANCED_INTENT_PARSER_AVAILABLE = True
except ImportError:
    EnhancedIntentParser = None
    ENHANCED_INTENT_PARSER_AVAILABLE = False

try:
    from .ml_malware_detector import MLMalwareDetector
    ML_MALWARE_DETECTOR_AVAILABLE = True
except ImportError:
    MLMalwareDetector = None
    ML_MALWARE_DETECTOR_AVAILABLE = False

__all__ = [
    'IntentParser',
    'RLAgent',
    'CommandGenerator'
]

# Add optional components if available
if ENHANCED_INTENT_PARSER_AVAILABLE:
    __all__.append('EnhancedIntentParser')

if ML_MALWARE_DETECTOR_AVAILABLE:
    __all__.append('MLMalwareDetector')
