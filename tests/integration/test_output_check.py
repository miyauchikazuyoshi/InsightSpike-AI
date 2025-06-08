#!/usr/bin/env python3
"""Test individual CLI command output."""

import subprocess
import sys

def test_list_command():
    """Test the list command specifically."""
    try:
        # Run the list command and capture output
        result = subprocess.run(
            [sys.executable, "-m", "insightspike.cli.deps_typer", "list"],
            capture_output=True,
            text=True,
            cwd="/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI"
        )
        
        print("=== LIST COMMAND TEST ===")
        print(f"Return code: {result.returncode}")
        print(f"STDOUT length: {len(result.stdout)}")
        print(f"STDERR length: {len(result.stderr)}")
        
        if result.stdout:
            print("STDOUT content:")
            print(result.stdout)
        else:
            print("No STDOUT content")
            
        if result.stderr:
            print("STDERR content:")
            print(result.stderr)
        else:
            print("No STDERR content")
            
    except Exception as e:
        print(f"Error running command: {e}")

def test_export_command():
    """Test the export-requirements command."""
    try:
        # Run the export-requirements command
        result = subprocess.run(
            [sys.executable, "-m", "insightspike.cli.deps_typer", "export-requirements", "--output", "test_output.txt"],
            capture_output=True,
            text=True,
            cwd="/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI"
        )
        
        print("\n=== EXPORT COMMAND TEST ===")
        print(f"Return code: {result.returncode}")
        print(f"STDOUT length: {len(result.stdout)}")
        print(f"STDERR length: {len(result.stderr)}")
        
        if result.stdout:
            print("STDOUT content:")
            print(result.stdout)
        else:
            print("No STDOUT content")
            
        if result.stderr:
            print("STDERR content:")
            print(result.stderr)
        else:
            print("No STDERR content")
            
        # Check if file was created
        import os
        if os.path.exists("test_output.txt"):
            print("✅ Output file was created")
            with open("test_output.txt", "r") as f:
                content = f.read()
                print(f"File content ({len(content)} chars):")
                print(content[:500] + "..." if len(content) > 500 else content)
            os.remove("test_output.txt")
        else:
            print("❌ Output file was NOT created")
            
    except Exception as e:
        print(f"Error running command: {e}")

if __name__ == "__main__":
    test_list_command()
    test_export_command()
