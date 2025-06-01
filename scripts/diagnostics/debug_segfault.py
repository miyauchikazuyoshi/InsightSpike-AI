#!/usr/bin/env python3
"""
Debug Segmentation Fault in Embedding Model Loading
==================================================

Systematic debugging of the segmentation fault occurring during
SentenceTransformer model loading with the current NumPy 1.x setup.
"""

import os
import sys
import traceback
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_dependency_versions():
    """Test and report dependency versions"""
    print("📋 Dependency Version Report")
    print("=" * 50)
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
    
    try:
        import faiss
        print(f"✅ FAISS: Available")
        # Test basic FAISS functionality
        test_index = faiss.IndexFlatL2(10)
        print(f"   Basic FAISS works: {test_index.ntotal}")
    except ImportError as e:
        print(f"❌ FAISS: {e}")
    except Exception as e:
        print(f"❌ FAISS error: {e}")
    
    try:
        import sentence_transformers
        print(f"✅ SentenceTransformers: {sentence_transformers.__version__}")
    except ImportError as e:
        print(f"❌ SentenceTransformers: {e}")
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers: {e}")

def test_minimal_sentence_transformer():
    """Test minimal SentenceTransformer loading"""
    print("\n🧪 Testing Minimal SentenceTransformer Loading")
    print("=" * 50)
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformer import successful")
        
        # Try with CPU explicitly
        print("⚠️ Attempting CPU-only model loading...")
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        print(f"✅ Model loaded successfully: {type(model)}")
        
        # Test encoding
        result = model.encode(["test sentence"])
        print(f"✅ Encoding successful: shape {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ SentenceTransformer test failed: {e}")
        traceback.print_exc()
        return False

def test_fallback_embedder():
    """Test the fallback embedder system"""
    print("\n🔄 Testing Fallback Embedder")
    print("=" * 50)
    
    try:
        from insightspike.utils.embedder import FallbackEmbedder
        
        embedder = FallbackEmbedder(384)
        result = embedder.encode(["test sentence", "another test"])
        print(f"✅ Fallback embedder works: shape {result.shape}")
        
        # Test reproducibility
        result2 = embedder.encode(["test sentence"])
        print(f"✅ Reproducible: {result[0][:3]} == {result2[0][:3]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback embedder failed: {e}")
        traceback.print_exc()
        return False

def test_safe_mode():
    """Test safe mode with environment variable"""
    print("\n🛡️ Testing Safe Mode")
    print("=" * 50)
    
    try:
        # Set safe mode
        os.environ['INSIGHTSPIKE_SAFE_MODE'] = '1'
        
        # Clear any cached modules
        modules_to_clear = [
            'insightspike.utils.embedder',
            'insightspike.embedder'
        ]
        
        for module in modules_to_clear:
            if module in sys.modules:
                del sys.modules[module]
        
        from insightspike.utils.embedder import get_model
        model = get_model()
        print(f"✅ Safe mode model loaded: {type(model)}")
        
        result = model.encode(["test sentence"])
        print(f"✅ Safe mode encoding: shape {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Safe mode failed: {e}")
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if 'INSIGHTSPIKE_SAFE_MODE' in os.environ:
            del os.environ['INSIGHTSPIKE_SAFE_MODE']

def main():
    """Main diagnostic function"""
    print("🔍 InsightSpike-AI Segmentation Fault Diagnostics")
    print("=" * 60)
    
    # Test dependency versions
    test_dependency_versions()
    
    # Test fallback system first (safest)
    fallback_ok = test_fallback_embedder()
    
    # Test safe mode
    safe_mode_ok = test_safe_mode()
    
    # Only test real SentenceTransformer if safe tests pass
    if fallback_ok and safe_mode_ok:
        print("\n⚠️ WARNING: The next test may cause a segmentation fault!")
        print("This will test the actual SentenceTransformer loading.")
        
        response = input("Continue with potentially dangerous test? (y/N): ")
        if response.lower() == 'y':
            st_ok = test_minimal_sentence_transformer()
        else:
            st_ok = False
            print("Skipped SentenceTransformer test")
    else:
        st_ok = False
        print("Skipping SentenceTransformer test due to safe test failures")
    
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"✅ Fallback Embedder: {'PASS' if fallback_ok else 'FAIL'}")
    print(f"✅ Safe Mode: {'PASS' if safe_mode_ok else 'FAIL'}")
    print(f"⚠️  SentenceTransformer: {'PASS' if st_ok else 'FAIL/SKIPPED'}")
    
    if fallback_ok and safe_mode_ok:
        print("\n💡 RECOMMENDATION: Use safe mode for deployment")
        print("   Set INSIGHTSPIKE_SAFE_MODE=1 to avoid segmentation faults")
    else:
        print("\n❌ CRITICAL: Basic fallback systems are failing")
        print("   Check Python environment and dependencies")

if __name__ == "__main__":
    main()
