#!/usr/bin/env python3
"""
Test if LocalProvider works with DistilGPT2.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from insightspike.implementations.layers.layer4_llm_interface import L4LLMInterface, LLMConfig


def test_local_llm():
    """Test LocalProvider initialization and generation."""
    
    print("=== Testing LocalProvider with DistilGPT2 ===")
    
    # Create config
    config = LLMConfig.from_provider(
        'local',
        model_name='distilgpt2',
        max_tokens=50,
        temperature=0.7
    )
    
    print(f"Config: {config}")
    
    # Initialize provider
    print("\nInitializing LocalProvider...")
    try:
        provider = L4LLMInterface(config)
        if provider.initialize():
            print("✓ Provider initialized successfully")
        else:
            print("✗ Provider initialization failed")
            return
    except Exception as e:
        print(f"✗ Error initializing provider: {e}")
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
            print(f"✓ Response: {result['response'][:100]}...")
        else:
            print(f"✗ Generation failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"✗ Error during generation: {e}")


if __name__ == "__main__":
    test_local_llm()