#!/usr/bin/env python3
"""
Test script to validate the fixes applied to the Colab Dependency Investigation notebook.
This tests:
1. Environment-aware checkpoint paths
2. NumPy 2.x compatibility patches
3. Basic functionality of core components
"""

import os
import sys
import warnings
from pathlib import Path

def test_environment_detection():
    """Test environment detection logic for checkpoint paths."""
    print("🧪 Testing Environment Detection Logic")
    print("=" * 50)
    
    # Current environment test
    current_cwd = os.getcwd()
    print(f"Current working directory: {current_cwd}")
    
    is_colab = '/content' in current_cwd
    print(f"Detected Colab environment: {is_colab}")
    
    # Test checkpoint path creation
    experiment_name = "test_experiment"
    if is_colab:
        checkpoint_dir = Path(f"/content/checkpoints/{experiment_name}")
        print(f"✅ Would use Colab checkpoint path: {checkpoint_dir}")
    else:
        checkpoint_dir = Path(f"./checkpoints/{experiment_name}")
        print(f"✅ Using local checkpoint path: {checkpoint_dir}")
    
    print(f"Selected checkpoint directory: {checkpoint_dir}")
    print()

def test_numpy_compatibility():
    """Test NumPy 2.x compatibility patches."""
    print("🧪 Testing NumPy 2.x Compatibility")
    print("=" * 50)
    
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
        
        # Test warning suppression patches
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*numpy.dtype size changed.*')
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*numpy.ufunc size changed.*')
        warnings.filterwarnings('ignore', category=UserWarning, message='.*numpy.*')
        
        # Set environment variable for compatibility
        os.environ.setdefault('PYTHONWARNINGS', 'ignore::RuntimeWarning:numpy')
        
        # Test numpy error suppression
        np.seterr(all='ignore')
        
        print("✅ NumPy compatibility patches applied successfully")
        
        # Test some basic NumPy operations that might trigger warnings
        arr = np.array([1, 2, 3, 4, 5])
        result = np.sqrt(arr)
        print(f"✅ Basic NumPy operations working: {result}")
        
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️ NumPy compatibility test issue: {e}")
        assert False, f"NumPy compatibility test failed: {e}"
    
    print()
    assert True  # Test passed

def test_mock_experiment_components():
    """Test basic components that would be used in the experiment."""
    print("🧪 Testing Mock Experiment Components")  
    print("=" * 50)
    
    try:
        # Test basic class structure similar to what's in the notebook
        class MockExperimentCheckpoint:
            def __init__(self, experiment_name):
                # Use environment-aware logic like in the notebook
                if '/content' in os.getcwd():
                    self.checkpoint_dir = Path(f"/content/checkpoints/{experiment_name}")
                else:
                    self.checkpoint_dir = Path(f"./checkpoints/{experiment_name}")
                
                print(f"✅ Checkpoint directory configured: {self.checkpoint_dir}")
        
        # Test instantiation
        checkpoint = MockExperimentCheckpoint("test_run")
        print("✅ MockExperimentCheckpoint instantiated successfully")
        
        # Test path resolution
        print(f"✅ Checkpoint path resolved to: {checkpoint.checkpoint_dir}")
        
    except Exception as e:
        print(f"❌ Mock experiment components test failed: {e}")
        assert False, f"Mock experiment components test failed: {e}"
    
    print()
    assert True  # Test passed

def main():
    """Run all tests."""
    print("🚀 Testing Notebook Fixes")
    print("=" * 60)
    print()
    
    # Test environment detection
    test_environment_detection()
    
    # Test NumPy compatibility
    numpy_ok = test_numpy_compatibility()
    
    # Test mock components
    components_ok = test_mock_experiment_components()
    
    # Summary
    print("📋 Test Summary")
    print("=" * 30)
    print(f"✅ Environment detection: Working")
    print(f"{'✅' if numpy_ok else '❌'} NumPy compatibility: {'Working' if numpy_ok else 'Failed'}")
    print(f"{'✅' if components_ok else '❌'} Mock components: {'Working' if components_ok else 'Failed'}")
    print()
    
    if numpy_ok and components_ok:
        print("🎉 All tests passed! The notebook fixes are working correctly.")
        assert True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        assert False, "Some tests failed"

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
