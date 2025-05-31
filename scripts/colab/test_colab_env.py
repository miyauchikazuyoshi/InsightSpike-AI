#!/usr/bin/env python3
"""
Colab faiss-gpu validation script
テスト用スクリプト: Colabでfaiss-gpuが正常に動作するか確認
"""

import sys
import os

def test_faiss_gpu():
    """Faiss-GPUの動作をテスト"""
    print("🔍 Testing faiss-gpu functionality...")
    
    try:
        import faiss
        print(f"✅ Faiss imported successfully, version: {faiss.__version__}")
        
        # CPU Indexのテスト
        dimension = 128
        n_vectors = 1000
        
        import numpy as np
        vectors = np.random.random((n_vectors, dimension)).astype('float32')
        
        # CPU Index
        index_cpu = faiss.IndexFlatL2(dimension)
        index_cpu.add(vectors)
        print(f"✅ CPU Index created with {index_cpu.ntotal} vectors")
        
        # GPU可用性テスト
        if hasattr(faiss, 'StandardGpuResources'):
            print("🚀 GPU resources available, testing GPU index...")
            
            try:
                # GPU Resourcesの作成
                gpu_res = faiss.StandardGpuResources()
                
                # CPU IndexをGPUに転送
                gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index_cpu)
                print(f"✅ GPU Index created with {gpu_index.ntotal} vectors")
                
                # 検索テスト
                query = np.random.random((1, dimension)).astype('float32')
                distances, indices = gpu_index.search(query, 5)
                print(f"✅ GPU search completed: found {len(indices[0])} results")
                
                return True
                
            except Exception as e:
                print(f"❌ GPU index creation failed: {e}")
                return False
        else:
            print("❌ GPU resources not available in this faiss installation")
            return False
            
    except ImportError as e:
        print(f"❌ Faiss import failed: {e}")
        return False

def test_torch_gpu():
    """PyTorchのGPU動作をテスト"""
    print("\n🔍 Testing PyTorch GPU functionality...")
    
    try:
        import torch
        print(f"✅ PyTorch imported, version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA device count: {torch.cuda.device_count()}")
            print(f"✅ Current device: {torch.cuda.current_device()}")
            print(f"✅ Device name: {torch.cuda.get_device_name(0)}")
            
            # 簡単なテンサー演算テスト
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.mm(x, y)
            print("✅ GPU tensor operations working")
            return True
        else:
            print("❌ CUDA not available")
            return False
            
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False

def test_sentence_transformers():
    """SentenceTransformersのテスト"""
    print("\n🔍 Testing SentenceTransformers...")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # テスト文の埋め込み
        sentences = ["This is a test sentence.", "Another test sentence."]
        embeddings = model.encode(sentences)
        print(f"✅ SentenceTransformers working, embedding shape: {embeddings.shape}")
        return True
        
    except ImportError as e:
        print(f"❌ SentenceTransformers import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ SentenceTransformers test failed: {e}")
        return False

def main():
    """メインテスト実行"""
    print("🚀 Colab Environment Validation for InsightSpike-AI")
    print("=" * 50)
    
    results = []
    
    # PyTorchテスト
    results.append(("PyTorch GPU", test_torch_gpu()))
    
    # Faiss-GPUテスト  
    results.append(("Faiss GPU", test_faiss_gpu()))
    
    # SentenceTransformersテスト
    results.append(("SentenceTransformers", test_sentence_transformers()))
    
    # 結果サマリー
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
        print("\n❌ Some tests failed. Please check the setup.")
        sys.exit(1)

if __name__ == "__main__":
    main()
