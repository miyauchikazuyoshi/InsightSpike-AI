#!/usr/bin/env python3
"""
Test script for the dependency management CLI
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from insightspike.cli.deps_typer import deps_app

if __name__ == "__main__":
    # Test the deps CLI directly
    deps_app()
