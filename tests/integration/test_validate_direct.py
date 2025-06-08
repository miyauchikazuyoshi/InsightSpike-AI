#!/usr/bin/env python3
"""Direct test of the validate function."""

import sys
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from insightspike.cli.deps_typer import validate

def test_validate_function():
    """Test the validate function directly."""
    try:
        print("=== CALLING validate DIRECTLY ===")
        
        # Call the function directly with default parameters
        validate(project_path=None)
        
        print("=== FUNCTION CALL COMPLETED ===")
        
    except Exception as e:
        print(f"❌ Error calling validate: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_validate_function()
