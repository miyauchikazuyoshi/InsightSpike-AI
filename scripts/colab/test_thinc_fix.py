#!/usr/bin/env python3
"""
Test script to verify thinc/numpy dependency conflict resolution
Checks that no thinc dependencies are installed after setup
"""

import subprocess
import sys
from pathlib import Path

def check_package_installed(package_name):
    """Check if a package is installed"""
    try:
        result = subprocess.run([sys.executable, "-c", f"import {package_name}"], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False

def get_package_version(package_name):
    """Get installed package version"""
    try:
        result = subprocess.run([sys.executable, "-c", f"import {package_name}; print({package_name}.__version__)"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None

def check_pip_list():
    """Check pip list for problematic packages"""
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                              capture_output=True, text=True)
        return result.stdout
    except Exception:
        return ""

def main():
    print("🔍 Testing thinc/numpy dependency conflict resolution")
    print("=" * 60)
    
    # Check critical packages
    packages_to_check = {
        "torch": "PyTorch",
        "faiss": "FAISS", 
        "numpy": "NumPy",
        "datasets": "Datasets",
        "transformers": "Transformers",
        "sentence_transformers": "Sentence Transformers"
    }
    
    print("\n📦 Checking critical packages:")
    for pkg, name in packages_to_check.items():
        if check_package_installed(pkg):
            version = get_package_version(pkg)
            print(f"✅ {name}: {version}")
        else:
            print(f"❌ {name}: Not installed")
    
    # Check for problematic packages
    print("\n⚠️ Checking for problematic packages:")
    problematic_packages = ["thinc", "spacy"]
    
    pip_output = check_pip_list()
    
    for pkg in problematic_packages:
        if check_package_installed(pkg):
            version = get_package_version(pkg)
            print(f"⚠️ {pkg.upper()}: {version} (potential conflict source)")
        elif pkg in pip_output.lower():
            print(f"⚠️ {pkg.upper()}: Found in pip list (potential conflict)")
        else:
            print(f"✅ {pkg.upper()}: Not installed (good)")
    
    # Check numpy version specifically
    print("\n🔢 NumPy version analysis:")
    numpy_version = get_package_version("numpy")
    if numpy_version:
        version_parts = numpy_version.split('.')
        if len(version_parts) >= 2:
            major, minor = int(version_parts[0]), int(version_parts[1])
            if major >= 2:
                print(f"✅ NumPy {numpy_version}: Compatible with thinc requirements")
            elif major == 1 and minor >= 24:
                print(f"✅ NumPy {numpy_version}: Compatible version")
            else:
                print(f"⚠️ NumPy {numpy_version}: May have compatibility issues")
        else:
            print(f"⚠️ NumPy version format unclear: {numpy_version}")
    else:
        print("❌ NumPy: Not available")
    
    # Test basic functionality
    print("\n🧪 Testing basic functionality:")
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    except Exception as e:
        print(f"❌ PyTorch: {e}")
    
    try:
        import datasets
        print(f"✅ Datasets: Available")
    except Exception as e:
        print(f"❌ Datasets: {e}")
    
    try:
        import faiss
        print(f"✅ FAISS: Available")
    except Exception as e:
        print(f"❌ FAISS: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Dependency conflict test completed!")

if __name__ == "__main__":
    main()
