#!/usr/bin/env python3
"""
Ultra-minimal bypass test to verify functionality
"""

import os
os.environ['INSIGHTSPIKE_LITE_MODE'] = '1'

print("🚀 MINIMAL BYPASS TEST")
print("=" * 30)

try:
    import sys
    sys.path.insert(0, 'src')
    
    # Test 1: Mock embedder
    print("1. Creating FallbackEmbedder...")
    from insightspike.utils.embedder import FallbackEmbedder
    embedder = FallbackEmbedder(dim=384)
    print("✅ FallbackEmbedder created")
    
    # Test 2: Simple encoding
    print("2. Testing encoding...")
    import numpy as np
    result = embedder.encode(['test'])
    print(f"✅ Encoding successful: {result.shape}")
    
    # Test 3: get_model bypass
    print("3. Testing get_model()...")
    from insightspike.utils.embedder import get_model  
    model = get_model()
    print(f"✅ get_model(): {type(model).__name__}")
    
    print("\n🎯 BYPASS TEST: PASSED ✅")
    
except Exception as e:
    print(f"\n❌ BYPASS TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
