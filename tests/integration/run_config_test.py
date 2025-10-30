#!/usr/bin/env python3
"""Quick config test runner."""

import sys
import os

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run
from tests.integration.test_simple_config_check import main

if __name__ == "__main__":
    main()