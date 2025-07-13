#!/usr/bin/env python3
"""
Test script to check for missing dependencies in the pyproject.toml
Identifies imports used in the codebase vs dependencies listed in pyproject.toml
"""

import importlib
import sys
from pathlib import Path


def test_dependency(module_name, description):
    """Test if a dependency can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name}: Available - {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: Missing - {description}")
        print(f"   Error: {e}")
        return False


def main():
    print("🔍 Testing Missing Dependencies")
    print("=" * 50)

    # Test the main missing dependency we identified
    dependencies_to_test = [
        ("torch_geometric", "PyTorch Geometric - Graph Neural Network components"),
        ("torch_geometric.data", "PyTorch Geometric Data structures"),
        ("torch_geometric.nn", "PyTorch Geometric Neural Network layers"),
        ("torch_geometric.utils", "PyTorch Geometric Utilities"),
        # Test other core dependencies that should be available
        ("numpy", "NumPy - Core numerical computing"),
        ("torch", "PyTorch - Deep learning framework"),
        ("transformers", "Hugging Face Transformers"),
        ("sentence_transformers", "Sentence Transformers for embeddings"),
        ("networkx", "NetworkX - Graph processing"),
        ("typer", "Typer - CLI framework"),
        ("click", "Click - CLI utilities"),
        ("rich", "Rich - Terminal formatting"),
        ("scipy", "SciPy - Scientific computing"),
        ("sklearn", "scikit-learn - Machine learning"),
        ("datasets", "Hugging Face Datasets"),
        ("pyyaml", "PyYAML - YAML processing"),
        ("psutil", "psutil - System information"),
        # Optional dependencies
        ("faiss", "FAISS - Vector similarity search (optional)"),
        ("spacy", "spaCy - Natural language processing (optional)"),
    ]

    print("Testing identified missing dependencies:")
    print("-" * 40)

    missing_count = 0
    available_count = 0

    for module, description in dependencies_to_test:
        if test_dependency(module, description):
            available_count += 1
        else:
            missing_count += 1
        print()

    print("=" * 50)
    print(f"📊 Summary:")
    print(f"   ✅ Available: {available_count}")
    print(f"   ❌ Missing: {missing_count}")
    print(f"   📋 Total tested: {len(dependencies_to_test)}")

    if missing_count > 0:
        print(f"\n🔧 Next Steps:")
        print(f"   1. Run 'poetry install' to install dependencies from pyproject.toml")
        print(f"   2. For torch-geometric, may need platform-specific installation")
        print(f"   3. For FAISS, may need separate installation (CPU vs GPU version)")
    else:
        print(f"\n🎉 All dependencies are available!")


if __name__ == "__main__":
    main()
