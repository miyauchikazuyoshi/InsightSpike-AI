#!/usr/bin/env python3
"""Direct test of the export-requirements function."""

import sys
from pathlib import Path
import os

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from insightspike.cli.deps_typer import export

def test_export_function():
    """Test the export-requirements function directly."""
    try:
        print("=== CALLING export_requirements DIRECTLY ===")
        
        output_file = Path("test_direct_export.txt")
        
        # Remove file if it exists
        if output_file.exists():
            output_file.unlink()
        
        # Call the function directly
        export(
            format="requirements",
            output=str(output_file)
        )
        
        print("=== FUNCTION CALL COMPLETED ===")
        
        # Check if file was created
        if output_file.exists():
            print(f"✅ Output file created: {output_file}")
            with open(output_file, "r") as f:
                content = f.read()
                print(f"File content ({len(content)} chars):")
                print(content[:500] + "..." if len(content) > 500 else content)
            # Clean up
            output_file.unlink()
        else:
            print("❌ Output file was NOT created")
        
    except Exception as e:
        print(f"❌ Error calling export_requirements: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_export_function()
