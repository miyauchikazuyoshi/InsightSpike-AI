#!/usr/bin/env python3
"""
Safe CLI Command Testing
========================

Test CLI commands without triggering the LLM model loading segmentation fault.
"""

import sys
from pathlib import Path
import subprocess

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_config_info():
    """Test config-info command"""
    print("Testing config-info command...")
    try:
        result = subprocess.run([
            "poetry", "run", "insightspike", "config-info"
        ], 
        cwd=project_root,
        capture_output=True, 
        text=True, 
        timeout=10
        )
        
        if result.returncode == 0:
            print("✓ config-info command works")
            return True
        else:
            print(f"✗ config-info failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ config-info command timed out")
        return False
    except Exception as e:
        print(f"✗ config-info error: {e}")
        return False

def test_help_command():
    """Test help command"""
    print("Testing help command...")
    try:
        result = subprocess.run([
            "poetry", "run", "insightspike", "--help"
        ], 
        cwd=project_root,
        capture_output=True, 
        text=True, 
        timeout=10
        )
        
        if result.returncode == 0 and "Commands" in result.stdout:
            print("✓ help command works")
            return True
        else:
            print(f"✗ help failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ help command timed out")
        return False
    except Exception as e:
        print(f"✗ help error: {e}")
        return False

def test_insights_command():
    """Test insights command (should be safe as it just shows registry)"""
    print("Testing insights command...")
    try:
        result = subprocess.run([
            "poetry", "run", "insightspike", "insights"
        ], 
        cwd=project_root,
        capture_output=True, 
        text=True, 
        timeout=15
        )
        
        if result.returncode == 0:
            print("✓ insights command works")
            return True
        else:
            print(f"✗ insights failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ insights command timed out")
        return False
    except Exception as e:
        print(f"✗ insights error: {e}")
        return False

if __name__ == "__main__":
    print("=== Safe CLI Command Testing ===")
    
    tests = [
        test_help_command,
        test_config_info,
        test_insights_command,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n=== Results ===")
    print(f"Help command: {'PASS' if results[0] else 'FAIL'}")
    print(f"Config info: {'PASS' if results[1] else 'FAIL'}")
    print(f"Insights command: {'PASS' if results[2] else 'FAIL'}")
    
    if all(results):
        print("\n🎉 All safe CLI commands work! Configuration issues resolved.")
        print("✓ CLI can access config without 'llm' attribute errors")
        print("✓ Basic commands execute successfully")
        print("⚠️ Note: LLM model loading still causes segmentation faults")
        sys.exit(0)
    else:
        print("\n❌ Some CLI commands still failing")
        sys.exit(1)
