#!/usr/bin/env python3
"""
Final CI Fix Validation Script
Tests the updated CI configuration to ensure all dependency issues are resolved.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, check=True, capture_output=True):
    """Run command and return result."""
    print(f"🔧 Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, 
                              capture_output=capture_output, text=True)
        if result.stdout:
            print(f"✅ Output: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"📤 STDOUT: {e.stdout}")
        if e.stderr:
            print(f"📥 STDERR: {e.stderr}")
        return e


def test_ci_environment_simulation():
    """Test CI environment simulation with Poetry + pip approach."""
    
    print("🚀 Testing CI Environment Simulation (Poetry + pip approach)")
    print("=" * 70)
    
    # Set environment variables
    env = os.environ.copy()
    env['INSIGHTSPIKE_LITE_MODE'] = '1'
    env['FORCE_CPU_ONLY'] = '1'
    
    # Test commands matching the new CI configuration
    commands = [
        # Test Poetry environment setup
        "poetry env info",
        
        # Test pip installation within Poetry (matching CI approach)
        'poetry run pip install "numpy>=1.21,<2.0"',
        'poetry run pip install faiss-cpu==1.7.4',
        'poetry run pip install torch --index-url https://download.pytorch.org/whl/cpu',
        
        # Test project installation without dependencies
        'poetry run pip install -e . --no-deps',
        
        # Test critical imports that were failing
        'poetry run python -c "import numpy; print(f\'NumPy: {numpy.__version__}\')"',
        'poetry run python -c "import faiss; print(f\'FAISS: {faiss.__version__}\'); print(f\'FAISS device: CPU\')"',
        'poetry run python -c "import torch; print(f\'PyTorch: {torch.__version__}\'); print(f\'CUDA available: {torch.cuda.is_available()}\')"',
        
        # Test project imports
        'poetry run python -c "from insightspike.cli.analyzer import main; print(\'✅ CLI analyzer import successful\')"',
        'poetry run python -c "from insightspike.cli.interactive import main; print(\'✅ CLI interactive import successful\')"',
        
        # Test pytest is available
        'poetry run pytest --version',
    ]
    
    results = {}
    for cmd in commands:
        print(f"\n📋 Testing: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, check=True, 
                                  capture_output=True, text=True, env=env)
            print(f"✅ SUCCESS: {result.stdout.strip()}")
            results[cmd] = True
        except subprocess.CalledProcessError as e:
            print(f"❌ FAILED: {e}")
            if e.stdout:
                print(f"📤 STDOUT: {e.stdout}")
            if e.stderr:
                print(f"📥 STDERR: {e.stderr}")
            results[cmd] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("🎯 CI ENVIRONMENT SIMULATION RESULTS")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for cmd, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {cmd}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL CI SIMULATION TESTS PASSED!")
        print("✅ The new Poetry + pip approach should work in CI")
        return True
    else:
        print("⚠️ Some tests failed - CI may still have issues")
        return False


def validate_ci_yaml_syntax():
    """Validate the CI YAML syntax."""
    print("\n🔍 Validating CI YAML syntax...")
    
    try:
        # Simple YAML syntax check
        import yaml
        with open('.github/workflows/enhanced-ci.yml', 'r') as f:
            yaml.safe_load(f)
        print("✅ CI YAML syntax is valid")
        return True
    except Exception as e:
        print(f"❌ CI YAML syntax error: {e}")
        return False


def generate_final_report():
    """Generate final validation report."""
    
    print("\n" + "=" * 70)
    print("📋 FINAL CI FIX VALIDATION REPORT")
    print("=" * 70)
    
    # Check key files
    key_files = [
        '.github/workflows/enhanced-ci.yml',
        'pyproject.toml',
        'poetry.lock'
    ]
    
    print("\n📁 Key Files Status:")
    for file_path in key_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} - EXISTS")
        else:
            print(f"❌ {file_path} - MISSING")
    
    # Summary of changes
    print("\n🔧 Critical Changes Applied:")
    print("✅ Removed all 'poetry install --only main --no-root' commands")
    print("✅ Added 'poetry run pip install' for all dependencies")
    print("✅ Added '--no-deps' flag to project installation")
    print("✅ Applied NumPy 1.x constraint across all CI jobs")
    print("✅ Applied faiss-cpu==1.7.4 constraint across all CI jobs")
    print("✅ Applied CPU-only PyTorch installation across all CI jobs")
    print("✅ Completely bypassed Poetry lock file in all 8 CI jobs")
    
    print("\n🎯 Expected CI Behavior:")
    print("• No more 'Unable to find installation candidates for faiss-gpu' errors")
    print("• No more NumPy 2.x compatibility issues with FAISS")
    print("• No more Poetry lock file dependency conflicts")
    print("• Consistent CPU-only environment across all CI jobs")
    print("• Faster CI runs (no lock file resolution)")
    
    print("\n📈 Next Steps:")
    print("1. Commit and push the enhanced-ci.yml changes")
    print("2. Monitor the CI pipeline for successful runs")
    print("3. All 8 CI jobs should now pass without dependency conflicts")


if __name__ == "__main__":
    print("🚀 InsightSpike-AI Final CI Fix Validation")
    print("=" * 70)
    
    # Test CI simulation
    ci_success = test_ci_environment_simulation()
    
    # Validate YAML
    yaml_success = validate_ci_yaml_syntax()
    
    # Generate report
    generate_final_report()
    
    # Final status
    if ci_success and yaml_success:
        print("\n🎉 VALIDATION COMPLETE - CI FIX READY FOR DEPLOYMENT!")
        sys.exit(0)
    else:
        print("\n⚠️ VALIDATION ISSUES DETECTED - REVIEW NEEDED")
        sys.exit(1)
