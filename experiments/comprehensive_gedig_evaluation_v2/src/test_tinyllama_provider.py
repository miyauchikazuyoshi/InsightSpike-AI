#!/usr/bin/env python3
"""
Test TinyLlama with LocalProvider directly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from insightspike.implementations.layers.layer4_llm_interface import L4LLMInterface, LLMConfig


def test_tinyllama_provider():
    """Test TinyLlama via LocalProvider."""
    
    print("=== Testing TinyLlama with LocalProvider ===")
    
    # Create config for TinyLlama
    config = LLMConfig.from_provider(
        'local',
        model_name='tinyllama',  # This should be converted to full model name
        max_tokens=100,
        temperature=0.7
    )
    
    print(f"Config: provider={config.provider}, model={config.model_name}")
    
    # Initialize provider
    print("\nInitializing LocalProvider with TinyLlama...")
    try:
        provider = L4LLMInterface(config)
        if provider.initialize():
            print("✓ Provider initialized successfully")
        else:
            print("✗ Provider initialization failed")
            return
    except Exception as e:
        print(f"✗ Error initializing provider: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test generation
    print("\nTesting generation...")
    try:
        result = provider.generate(
            context={'retrieved_documents': []},
            question="What is 2 + 2?"
        )
        print(f"Result: {result}")
        
        if result.get('success'):
            print(f"✓ Response: {result['response'][:200]}...")
        else:
            print(f"✗ Generation failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_tinyllama_provider()