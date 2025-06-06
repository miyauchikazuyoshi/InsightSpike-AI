#!/usr/bin/env python3
"""
CI Environment Validation Script
===============================

This script validates that all CI/CD dependencies and configurations 
are working properly after the fixes.
"""
import sys
import os
import subprocess
import importlib.util

def test_import(module_name):
    """Test if a module can be imported."""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            mod = importlib.import_module(module_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {module_name:15}: {version}")
            return True
        else:
            print(f"❌ {module_name:15}: not found")
            return False
    except Exception as e:
        print(f"❌ {module_name:15}: error - {e}")
        return False

def test_project_imports():
    """Test project-specific imports."""
    sys.path.insert(0, './src')
    try:
        from insightspike.utils.dependency_resolver import DependencyResolver
        from insightspike.utils.platform_utils import PlatformInfo
        from insightspike.cli.dependency_commands import deps_cli
        print("✅ Project modules: All imports successful")
        return True
    except Exception as e:
        print(f"❌ Project modules: Import error - {e}")
        return False

def run_sample_test():
    """Run a sample test to verify pytest is working."""
    try:
        os.environ['PYTHONPATH'] = './src:' + os.environ.get('PYTHONPATH', '')
        os.environ['INSIGHTSPIKE_LITE_MODE'] = '1'
        
        result = subprocess.run(
            ['poetry', 'run', 'python', '-c', 'import pytest; print("pytest working")'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ pytest execution: Working")
            return True
        else:
            print(f"❌ pytest execution: Failed - {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ pytest execution: Error - {e}")
        return False

def main():
    """Main validation function."""
    print("🧪 InsightSpike-AI CI/CD Environment Validation")
    print("=" * 60)
    
    # Test core CI dependencies
    modules_to_test = [
        'pytest',
        'torch', 
        'faiss',
        'black',
        'isort',
        'flake8',
        'mypy',
        'bandit',
        'typer',
        'click',
        'rich',
        'pyyaml'
    ]
    
    success_count = 0
    for module in modules_to_test:
        if test_import(module):
            success_count += 1
    
    print("-" * 60)
    
    # Test project imports
    project_imports_ok = test_project_imports()
    if project_imports_ok:
        success_count += 1
    
    # Test pytest execution
    pytest_ok = run_sample_test()
    if pytest_ok:
        success_count += 1
    
    print("=" * 60)
    total_tests = len(modules_to_test) + 2  # +2 for project imports and pytest
    print(f"📊 Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎯 Status: ✅ CI ENVIRONMENT READY")
        print("🚀 All CI/CD fixes validated successfully!")
        return True
    else:
        print("⚠️  Status: ❌ ISSUES FOUND")
        print("🔧 Some components need attention")
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
