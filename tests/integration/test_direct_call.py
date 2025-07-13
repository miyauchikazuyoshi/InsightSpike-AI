#!/usr/bin/env python3
"""Direct test of the list function to see if there are import or Rich issues."""

import sys
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from insightspike.cli.deps_typer import list


def test_list_function():
    """Test the list function directly."""
    try:
        print("=== CALLING list_deps DIRECTLY ===")

        # Call the function directly with default parameters
        list(
            platform=None,
            include_groups=False,
            platform_specific_only=False,
            json_output=False,
        )

        print("=== FUNCTION CALL COMPLETED ===")

    except Exception as e:
        print(f"❌ Error calling list_deps: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_list_function()
