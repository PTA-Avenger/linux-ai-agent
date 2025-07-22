#!/usr/bin/env python3
"""
Linux AI Agent - Main Entry Point

A modular, Python-based AI agent for Linux that performs file operations,
system monitoring, malware detection using ClamAV, and heuristic scanning
using entropy analysis.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from interface import CLI
from utils import get_logger


def main():
    """Main entry point for the Linux AI Agent."""
    logger = get_logger("main")
    
    try:
        logger.info("Starting Linux AI Agent")
        
        # Create and run CLI
        cli = CLI()
        cli.run()
        
        logger.info("Linux AI Agent shutdown complete")
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
