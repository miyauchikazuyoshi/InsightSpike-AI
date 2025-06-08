#!/usr/bin/env python3
"""
Quick test of bypass mechanisms after directory reorganization
"""

import os
import sys

# Set environment variables
os.environ['INSIGHTSPIKE_LITE_MODE'] = '1'
os.environ['FORCE_CPU_ONLY'] = '1'

# Add src to path
sys.path.insert(0, 'src')

def test_bypass_mechanisms():
    print("🔍 Testing Bypass Mechanisms After Directory Reorganization")
    print("=" * 60)
    
    print(f"INSIGHTSPIKE_LITE_MODE: {os.getenv('INSIGHTSPIKE_LITE_MODE')}")
    print(f"FORCE_CPU_ONLY: {os.getenv('FORCE_CPU_ONLY')}")
    print()
    
    try:
        # Test 1: Direct FallbackEmbedder
        print("1. Testing FallbackEmbedder...")
        from insightspike.utils.embedder import FallbackEmbedder
        embedder = FallbackEmbedder(dim=384)
        result = embedder.encode(['Test sentence'])
        print(f"✅ FallbackEmbedder: {result.shape}")
        
        # Test 2: get_model function
        print("2. Testing get_model()...")
        from insightspike.utils.embedder import get_model
        model = get_model()
        print(f"✅ Model type: {type(model).__name__}")
        
        # Test 3: Model encoding
        print("3. Testing model encoding...")
        result2 = model.encode(['Another test'])
        print(f"✅ Encoding: {result2.shape}")
        
        # Test 4: Memory manager
        print("4. Testing L2MemoryManager...")
        from insightspike.core.layers.layer2_memory_manager import L2MemoryManager
        memory = L2MemoryManager(dim=result.shape[1])
        print(f"✅ Memory manager: dim={memory.dim}")
        
        print()
        print("🎯 STATUS: ALL BYPASS MECHANISMS WORKING ✅")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_bypass_mechanisms()
    exit(0 if success else 1)
