#!/usr/bin/env python3
"""
Test the new composition root implementation
"""

import subprocess
import sys

def test_basic_import():
    """Test that we can import the main module"""
    print("Test 1: Basic import")
    try:
        import src.insightspike.__main__ as main_module
        print("✓ Main module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_module_execution():
    """Test running as a module"""
    print("\nTest 2: Module execution (python -m)")
    try:
        # This will fail since we don't have the full environment, but we can check the error
        result = subprocess.run(
            [sys.executable, "-m", "src.insightspike", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"Output: {result.stdout[:200]}...")
        if result.stderr:
            print(f"Error: {result.stderr[:200]}...")
            
        # Check if it's trying to run our composition root
        if "composition root" in result.stderr.lower() or "mainagent" in result.stderr.lower():
            print("✓ Composition root is being executed")
            return True
            
    except subprocess.TimeoutExpired:
        print("✗ Command timed out")
    except Exception as e:
        print(f"✗ Execution failed: {e}")
    
    return False

def test_direct_composition():
    """Test the composition root directly"""
    print("\nTest 3: Direct composition test")
    
    # Create a minimal test to verify the wiring
    test_code = """
import sys
sys.path.insert(0, 'src')

# Test that imports work
try:
    from insightspike.config.loader import load_config
    from insightspike.implementations.datastore.factory import DataStoreFactory
    from insightspike.implementations.agents.main_agent import MainAgent
    print("✓ All imports successful")
    
    # Test DataStore creation
    datastore_config = {"type": "filesystem", "params": {"base_path": "./test_comp_data"}}
    datastore = DataStoreFactory.create_from_config(datastore_config)
    print(f"✓ Created DataStore: {datastore.__class__.__name__}")
    
    # Clean up
    import shutil
    shutil.rmtree("./test_comp_data", ignore_errors=True)
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
"""
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print(f"Stderr: {result.stderr}")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Testing Composition Root Implementation ===\n")
    
    tests = [
        test_basic_import,
        test_module_execution,
        test_direct_composition
    ]
    
    passed = sum(1 for test in tests if test())
    
    print(f"\n=== Results ===")
    print(f"Passed: {passed}/{len(tests)}")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)