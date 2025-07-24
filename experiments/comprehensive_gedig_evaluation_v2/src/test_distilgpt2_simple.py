#!/usr/bin/env python3
"""
Simple test with DistilGPT2.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from insightspike.implementations.layers.layer4_llm_interface import L4LLMInterface, LLMConfig


def test_distilgpt2():
    """Test DistilGPT2."""
    
    print("=== Testing DistilGPT2 ===")
    
    # Create config
    config = LLMConfig.from_provider(
        'local',
        model_name='distilgpt2',
        max_tokens=50,
        temperature=0.7
    )
    
    print(f"Config: provider={config.provider}, model={config.model_name}")
    
    # Initialize provider
    print("\nInitializing LocalProvider with DistilGPT2...")
    provider = L4LLMInterface(config)
    
    if provider.initialize():
        print("✓ Provider initialized successfully")
        
        # Test generation
        print("\nTesting generation...")
        result = provider.generate_response_detailed(
            context={'retrieved_documents': []},
            question="What is mathematics?"
        )
        
        if result.get('success'):
            print(f"✓ Response: {result['response'][:100]}...")
        else:
            print(f"✗ Generation failed: {result.get('error', 'Unknown')}")
    else:
        print("✗ Provider initialization failed")
    
    print("\n=== Test completed ===")


if __name__ == "__main__":
    test_distilgpt2()