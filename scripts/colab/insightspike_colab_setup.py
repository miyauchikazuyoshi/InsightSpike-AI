# InsightSpike-AI Optimized Colab Setup
# Generated from dependency investigation 2025-01-XX

import subprocess
import sys
import time
from datetime import datetime

def install_optimized_faiss():
    """Install FAISS with CUDA version awareness"""
    print("🔧 Installing optimized FAISS...")
    
    # Detect CUDA version
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            cuda_major = cuda_version.split('.')[0] if cuda_version else "unknown"
            
            if cuda_major == '12':
                packages = ["faiss-gpu==1.8.0+cu12", "faiss-gpu-cu12", "faiss-gpu"]
            elif cuda_major == '11':
                packages = ["faiss-gpu-cu11", "faiss-gpu"]
            else:
                packages = ["faiss-gpu"]
        else:
            packages = ["faiss-cpu"]
    except:
        packages = ["faiss-gpu", "faiss-cpu"]
    
    for package in packages:
        try:
            print(f"   Trying {package}...")
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', package, '--upgrade'], 
                                  capture_output=True, text=True, timeout=180)
            
            # Test if it works
            import faiss
            if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
                print(f"   ✅ {package} working with GPU")
                return True
            elif 'cpu' in package:
                print(f"   ✅ {package} working with CPU")
                return True
        except Exception as e:
            print(f"   ❌ {package} failed: {e}")
            continue
    
    return False

def setup_insightspike_environment():
    """Complete environment setup for InsightSpike-AI"""
    print("🚀 Setting up InsightSpike-AI environment...")
    start_time = time.time()
    
    # Install core dependencies
    core_packages = [
        "numpy>=2.0",
        "torch",
        "transformers",
        "datasets",
        "psutil",
        "GPUtil"
    ]
    
    for package in core_packages:
        print(f"📦 Installing {package}...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                      capture_output=True, text=True)
    
    # Install optimized FAISS
    faiss_success = install_optimized_faiss()
    
    setup_time = time.time() - start_time
    print(f"\n✅ Setup complete in {setup_time:.1f} seconds")
    print(f"🧠 FAISS: {'✅ Working' if faiss_success else '❌ Issues'}")
    
    return faiss_success

# Run setup if executed
if __name__ == "__main__":
    setup_insightspike_environment()
