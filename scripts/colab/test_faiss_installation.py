#!/usr/bin/env python3
"""
FAISS Installation Test Script
Tests the enhanced FAISS installation approach for Google Colab
"""

import subprocess
import sys
import time

def run_command(cmd, timeout=180):
    """Run a command with timeout"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Command timed out after {timeout} seconds"

def test_faiss_installation():
    """Test the enhanced FAISS installation process"""
    print("🧪 Testing Enhanced FAISS Installation Process")
    print("=" * 60)
    
    # Step 1: Install CUDA runtime dependencies
    print("\n📦 Step 1: Installing CUDA runtime libraries...")
    success, stdout, stderr = run_command(
        "pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12"
    )
    
    if success:
        print("✅ CUDA runtime libraries installed successfully")
    else:
        print("⚠️ CUDA runtime installation failed, will try CPU fallback")
        print(f"Error: {stderr}")
    
    # Step 2: Install FAISS-GPU-CU12
    if success:
        print("\n🔍 Step 2: Installing FAISS-GPU-CU12...")
        success, stdout, stderr = run_command(
            "pip install 'faiss-gpu-cu12>=1.11.0'", 
            timeout=180
        )
        
        if success:
            print("✅ FAISS-GPU-CU12 installed successfully")
        else:
            print("⚠️ FAISS-GPU-CU12 failed, installing CPU fallback...")
            success, stdout, stderr = run_command("pip install faiss-cpu")
            if success:
                print("✅ FAISS-CPU installed successfully")
            else:
                print("❌ Both FAISS-GPU and FAISS-CPU installation failed")
                return False
    else:
        # Direct CPU fallback
        print("\n🔄 Step 2: Installing FAISS-CPU (fallback)...")
        success, stdout, stderr = run_command("pip install faiss-cpu")
        if success:
            print("✅ FAISS-CPU installed successfully")
        else:
            print("❌ FAISS-CPU installation failed")
            return False
    
    # Step 3: Test FAISS import and functionality
    print("\n🧪 Step 3: Testing FAISS functionality...")
    
    try:
        import faiss
        print(f"✅ FAISS imported successfully, version: {faiss.__version__}")
        
        # Test GPU availability
        try:
            gpu_count = faiss.get_num_gpus()
            print(f"🔍 GPU count detected: {gpu_count}")
            
            if gpu_count > 0:
                # Test GPU functionality
                res = faiss.StandardGpuResources()
                print("🚀 FAISS GPU support is available and working")
                
                # Quick GPU test
                import numpy as np
                dimension = 128
                nb = 1000
                nq = 10
                
                # Generate test data
                xb = np.random.random((nb, dimension)).astype('float32')
                xq = np.random.random((nq, dimension)).astype('float32')
                
                # Create index
                index = faiss.IndexFlatL2(dimension)
                gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
                gpu_index.add(xb)
                
                # Search
                D, I = gpu_index.search(xq, 5)
                print(f"✅ GPU search test completed: {len(I)} queries processed")
                
            else:
                print("⚠️ No GPUs detected - using CPU version")
                
        except Exception as e:
            print(f"⚠️ GPU functionality not available: {e}")
            print("💡 CPU-only FAISS is functional")
        
        # Test basic CPU functionality
        try:
            import numpy as np
            dimension = 64
            nb = 100
            
            # Generate test data
            xb = np.random.random((nb, dimension)).astype('float32')
            xq = np.random.random((1, dimension)).astype('float32')
            
            # Create CPU index
            index = faiss.IndexFlatL2(dimension)
            index.add(xb)
            
            # Search
            D, I = index.search(xq, 5)
            print(f"✅ CPU search test completed: found {len(I[0])} results")
            
        except Exception as e:
            print(f"❌ Basic FAISS functionality test failed: {e}")
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ FAISS import failed: {e}")
        return False

def main():
    """Main test function"""
    print("🔬 FAISS Installation Validation Test")
    print("This script tests the enhanced FAISS installation approach")
    print("for resolving Google Colab dependency issues.")
    print()
    
    start_time = time.time()
    success = test_faiss_installation()
    end_time = time.time()
    
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    print("\n" + "=" * 60)
    print(f"📊 Test completed in {minutes}m {seconds}s")
    
    if success:
        print("🎉 FAISS installation test PASSED")
        print("✅ The enhanced installation approach is working correctly")
    else:
        print("❌ FAISS installation test FAILED")
        print("🔧 Please check the error messages above for troubleshooting")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
