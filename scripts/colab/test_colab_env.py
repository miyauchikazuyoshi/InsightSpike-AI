#!/usr/bin/env python3
"""
Colab Environment Validation Script for InsightSpike-AI
Tests essential components: PyTorch GPU, FAISS, and SentenceTransformers
"""

import sys
import os

def test_faiss_gpu():
    """Test Faiss GPU functionality"""
    print("🔍 Testing FAISS functionality...")
    
    try:
        import faiss
        print(f"✅ FAISS imported successfully, version: {faiss.__version__}")
        
        # Test CPU Index first
        dimension = 128
        n_vectors = 1000
        
        import numpy as np
        vectors = np.random.random((n_vectors, dimension)).astype('float32')
        
        # CPU Index
        index_cpu = faiss.IndexFlatL2(dimension)
        index_cpu.add(vectors)
        print(f"✅ CPU Index created with {index_cpu.ntotal} vectors")
        
        # Test GPU availability
        if hasattr(faiss, 'StandardGpuResources'):
            print("🚀 GPU resources available, testing GPU index...")
            
            try:
                # Create GPU Resources
                gpu_res = faiss.StandardGpuResources()
                
                # Transfer CPU Index to GPU
                gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index_cpu)
                print(f"✅ GPU Index created with {gpu_index.ntotal} vectors")
                
                # Search test
                query = np.random.random((1, dimension)).astype('float32')
                distances, indices = gpu_index.search(query, 5)
                print(f"✅ GPU search completed: found {len(indices[0])} results")
                
                return True
                
            except Exception as e:
                print(f"⚠️ GPU index creation failed: {e}")
                print("💡 CPU-only FAISS will be used (slower but functional)")
                return True  # CPU-only is still acceptable
        else:
            print("⚠️ GPU resources not available in this FAISS installation")
            print("💡 CPU-only FAISS will be used (slower but functional)")
            return True  # CPU-only is still acceptable
            
    except ImportError as e:
        print(f"❌ FAISS import failed: {e}")
        print("🔧 Fix: Run setup script or install manually:")
        print("   !pip install faiss-cpu")
        print("   # OR for GPU support:")
        print("   !pip install faiss-gpu")
        return False

def test_torch_gpu():
    """Test PyTorch GPU functionality"""
    print("\n🔍 Testing PyTorch GPU functionality...")
    
    try:
        import torch
        print(f"✅ PyTorch imported, version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA device count: {torch.cuda.device_count()}")
            print(f"✅ Current device: {torch.cuda.current_device()}")
            print(f"✅ Device name: {torch.cuda.get_device_name(0)}")
            
            # Simple tensor operation test
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.mm(x, y)
            print("✅ GPU tensor operations working")
            return True
        else:
            print("⚠️ CUDA not available - using CPU mode")
            print("💡 Enable GPU: Runtime > Change runtime type > GPU")
            return True  # CPU mode is still functional
            
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        print("🔧 Fix: Run setup script or install manually:")
        print("   !pip install torch")
        return False

def test_sentence_transformers():
    """Test SentenceTransformers functionality"""
    print("\n🔍 Testing SentenceTransformers...")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test sentence embedding
        sentences = ["This is a test sentence.", "Another test sentence."]
        embeddings = model.encode(sentences)
        print(f"✅ SentenceTransformers working, embedding shape: {embeddings.shape}")
        return True
        
    except ImportError as e:
        print(f"❌ SentenceTransformers import failed: {e}")
        print("🔧 Fix: Run setup script or install manually:")
        print("   !pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"❌ SentenceTransformers test failed: {e}")
        return False

def main():
    """Main test execution"""
    print("🚀 Colab Environment Validation for InsightSpike-AI")
    print("=" * 50)
    
    results = []
    
    # PyTorch test
    results.append(("PyTorch GPU", test_torch_gpu()))
    
    # FAISS test  
    results.append(("FAISS", test_faiss_gpu()))
    
    # SentenceTransformers test
    results.append(("SentenceTransformers", test_sentence_transformers()))
    
    # Results summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Colab environment is ready for InsightSpike-AI")
        print("💡 You can now run:")
        print("   - !PYTHONPATH=src python -m insightspike.cli embed --path data/raw/test_sentences.txt")
        print("   - !PYTHONPATH=src python -m insightspike.cli graph")
        print("   - !PYTHONPATH=src python -m insightspike.cli loop 'Your question here'")
    else:
        failed_tests = [name for name, result in results if not result]
        print(f"\n⚠️ Some tests had issues: {', '.join(failed_tests)}")
        print("💡 Check the setup recommendations above and try:")
        print("   - Re-run the setup cell with a different SETUP_CHOICE")
        print("   - Use Debug Setup (3) for detailed installation logs")
        print("   - Check the troubleshooting section below")

if __name__ == "__main__":
    main()
