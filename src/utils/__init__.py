"""
Utilities package for Linux AI Agent.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../logs'))

from logger import get_logger, log_operation

__all__ = ['get_logger', 'log_operation']
