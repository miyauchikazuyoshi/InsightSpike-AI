#!/usr/bin/env python3
"""
Cross-Environment Testing Script
===============================

Tests InsightSpike-AI functionality across different environments:
- Local development
- Google Colab simulation
- CI/CD simulation
- Safe mode validation
"""

import os
import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_local_environment():
    """Test local development environment"""
    print("🏠 Testing Local Environment")
    print("=" * 50)
    
    try:
        # Test imports
        from insightspike.config import get_config
        from insightspike.utils.embedder import get_model, FallbackEmbedder
        from insightspike.core.layers.layer2_memory_manager import L2MemoryManager
        
        config = get_config()
        print(f"✅ Config loaded: {config.models.embedding_model}")
        
        # Test embedder
        embedder = FallbackEmbedder(384)
        result = embedder.encode(['Test message'])
        print(f"✅ Embedder works: shape {result.shape}")
        
        # Test memory manager
        memory = L2MemoryManager(dim=384)
        print(f"✅ Memory manager: {memory.dim} dimensions")
        
        return True
        
    except Exception as e:
        print(f"❌ Local environment test failed: {e}")
        traceback.print_exc()
        return False

def test_colab_simulation():
    """Test Google Colab simulation"""
    print("\n🌐 Testing Google Colab Simulation")
    print("=" * 50)
    
    try:
        # Simulate Colab environment
        os.environ["COLAB_GPU"] = "1"
        
        # Re-import to pick up environment changes
        from insightspike.config import reload_config, get_config
        reload_config()
        config = get_config()
        
        print(f"✅ Colab config: environment={config.environment}")
        print(f"✅ Device setting: {config.models.device}")
        
        # Test safe mode works in Colab
        os.environ["INSIGHTSPIKE_SAFE_MODE"] = "1"
        from insightspike.utils.embedder import FallbackEmbedder
        embedder = FallbackEmbedder(384)
        result = embedder.encode(['Colab test'])
        print(f"✅ Safe mode in Colab: shape {result.shape}")
        
        # Clean up environment
        del os.environ["COLAB_GPU"]
        del os.environ["INSIGHTSPIKE_SAFE_MODE"]
        
        return True
        
    except Exception as e:
        print(f"❌ Colab simulation failed: {e}")
        traceback.print_exc()
        return False

def test_dependency_compatibility():
    """Test dependency compatibility"""
    print("\n📦 Testing Dependency Compatibility")
    print("=" * 50)
    
    try:
        import numpy as np
        import torch
        
        print(f"✅ NumPy version: {np.__version__}")
        print(f"✅ PyTorch version: {torch.__version__}")
        
        # Check NumPy version compatibility
        if np.__version__.startswith('1.'):
            print("✅ NumPy 1.x compatibility confirmed")
        else:
            print(f"⚠️ NumPy version {np.__version__} may have compatibility issues")
        
        # Test FAISS availability
        try:
            import faiss
            print(f"✅ FAISS available: {faiss.__version__ if hasattr(faiss, '__version__') else 'unknown version'}")
        except ImportError:
            print("⚠️ FAISS not available - will use fallback")
        
        # Test sentence-transformers
        try:
            import sentence_transformers
            print(f"✅ Sentence Transformers: {sentence_transformers.__version__}")
        except ImportError:
            print("⚠️ Sentence Transformers not available - safe mode required")
        
        return True
        
    except Exception as e:
        print(f"❌ Dependency test failed: {e}")
        return False

def test_safe_mode_robustness():
    """Test safe mode robustness"""
    print("\n🛡️ Testing Safe Mode Robustness")
    print("=" * 50)
    
    try:
        # Enable safe mode
        os.environ["INSIGHTSPIKE_SAFE_MODE"] = "1"
        
        # Test all core components work in safe mode
        from insightspike.utils.embedder import FallbackEmbedder
        from insightspike.core.layers.layer2_memory_manager import L2MemoryManager, Episode
        
        # Test embedder
        embedder = FallbackEmbedder(384)
        embeddings = embedder.encode([
            "Safe mode test document 1",
            "Safe mode test document 2",
            "Safe mode test document 3"
        ])
        print(f"✅ Safe embeddings: {embeddings.shape}")
        
        # Test memory operations
        memory = L2MemoryManager(dim=384)
        episode1 = Episode(embeddings[0], "Document 1")
        episode2 = Episode(embeddings[1], "Document 2")
        memory.episodes = [episode1, episode2]
        print(f"✅ Safe memory ops: {len(memory.episodes)} episodes")
        
        # Test search functionality
        search_results = memory.search(embeddings[2], top_k=2)
        print(f"✅ Safe search: {len(search_results)} results")
        
        # Clean up
        del os.environ["INSIGHTSPIKE_SAFE_MODE"]
        
        return True
        
    except Exception as e:
        print(f"❌ Safe mode test failed: {e}")
        traceback.print_exc()
        return False

def test_end_to_end_workflow():
    """Test end-to-end workflow"""
    print("\n🔄 Testing End-to-End Workflow")
    print("=" * 50)
    
    try:
        # Enable safe mode for stability
        os.environ["INSIGHTSPIKE_SAFE_MODE"] = "1"
        
        from insightspike.utils.embedder import FallbackEmbedder
        from insightspike.core.layers.layer2_memory_manager import L2MemoryManager
        
        # Simulate document processing workflow
        documents = [
            "Artificial intelligence is transforming technology.",
            "Machine learning enables pattern recognition.",
            "Neural networks mimic brain functionality.",
            "Deep learning uses multiple hidden layers.",
            "Natural language processing understands text."
        ]
        
        # Step 1: Initialize components
        embedder = FallbackEmbedder(384)
        memory = L2MemoryManager(dim=384)
        print("✅ Components initialized")
        
        # Step 2: Process documents
        embeddings = embedder.encode(documents)
        print(f"✅ Documents embedded: {embeddings.shape}")
        
        # Step 3: Store in memory
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            success = memory.store_episode(doc, c_value=0.5 + (i * 0.1))
            if not success:
                print(f"⚠️ Failed to store episode {i}")
        print(f"✅ Episodes stored: {len(memory.episodes)}")
        
        # Step 4: Query system
        query = "What is artificial intelligence?"
        results = memory.search_episodes(query, k=3)
        print(f"✅ Search results: {len(results)} matches")
        
        # Step 5: Validate results
        for i, result in enumerate(results):
            score = result.get('weighted_score', 0)
            text = result.get('text', 'N/A')
            print(f"   {i+1}. Score: {score:.3f} - {text[:50]}...")
        
        # Clean up
        del os.environ["INSIGHTSPIKE_SAFE_MODE"]
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all cross-environment tests"""
    print("🔍 InsightSpike-AI Cross-Environment Testing")
    print("=" * 70)
    
    tests = [
        ("Local Environment", test_local_environment),
        ("Colab Simulation", test_colab_simulation),
        ("Dependency Compatibility", test_dependency_compatibility),
        ("Safe Mode Robustness", test_safe_mode_robustness),
        ("End-to-End Workflow", test_end_to_end_workflow)
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
    
    # Summary
    print("\n" + "=" * 70)
    print("🎯 CROSS-ENVIRONMENT TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status} {name}")
    
    print(f"\n📊 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Cross-environment compatibility confirmed!")
        return True
    else:
        print("⚠️ Some tests failed - Check individual test outputs")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
