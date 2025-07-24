#!/usr/bin/env python3
"""
Debug TinyLlama loading issue.
"""

import sys
import logging
from pathlib import Path

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from insightspike.implementations.layers.layer4_llm_interface import L4LLMInterface, LLMConfig


def test_tinyllama_minimal():
    """Minimal test for TinyLlama."""
    
    print("=== Testing TinyLlama Loading ===")
    
    # Step 1: Create config
    print("\n1. Creating config...")
    config = LLMConfig.from_provider(
        'local',
        model_name='tinyllama',
        max_tokens=50,
        temperature=0.7
    )
    print(f"   Config created: provider={config.provider}, model={config.model_name}")
    
    # Step 2: Create provider
    print("\n2. Creating provider...")
    provider = L4LLMInterface(config)
    print("   Provider created")
    
    # Step 3: Initialize
    print("\n3. Initializing provider (loading model)...")
    print("   This may take a minute on first run...")
    
    try:
        success = provider.initialize()
        if success:
            print("   ✓ Provider initialized successfully!")
        else:
            print("   ✗ Provider initialization failed")
            return
    except Exception as e:
        print(f"   ✗ Exception during initialization: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Test generation
    print("\n4. Testing generation...")
    try:
        result = provider.generate_response_detailed(
            context={'retrieved_documents': []},
            question="What is 2 + 2?"
        )
        
        print(f"   Success: {result.get('success', False)}")
        if result.get('success'):
            print(f"   Response: {result['response'][:100]}...")
        else:
            print(f"   Error: {result.get('error', 'Unknown')}")
            
    except Exception as e:
        print(f"   ✗ Exception during generation: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Test completed ===")


if __name__ == "__main__":
    test_tinyllama_minimal()