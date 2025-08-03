"""
Debug LLM to see what's happening
"""

import os
import sys
from pathlib import Path
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike import MainAgent
from insightspike.providers import ProviderFactory


def test_provider_directly():
    """Test LLM provider directly."""
    print("Testing LLM provider directly...")
    
    # Set API key
    os.environ["ANTHROPIC_API_KEY"] = "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"
    
    try:
        # Get provider
        provider = ProviderFactory.get_provider("anthropic", model="claude-3-opus-20240229")
        
        # Test simple prompt
        prompt = "What is 2 + 2? Please provide just the number."
        print(f"\nPrompt: {prompt}")
        
        response = provider.complete(prompt)
        print(f"Response: {response}")
        
        # Test with context
        prompt_with_context = """Context: Apples are fruits that grow on trees. They come in red, green, and yellow colors.

Question: What colors can apples be?

Answer:"""
        print(f"\nPrompt with context: {prompt_with_context}")
        
        response2 = provider.complete(prompt_with_context)
        print(f"Response: {response2}")
        
    except Exception as e:
        print(f"Error testing provider: {e}")
        import traceback
        traceback.print_exc()


def test_with_mock():
    """Test with mock provider to see if issue is with LLM."""
    print("\n\nTesting with mock provider...")
    
    config = {
        "llm": {
            "provider": "mock",
            "model": "mock"
        },
        "memory": {
            "max_retrieved_docs": 5
        }
    }
    
    agent = MainAgent(config)
    
    # Add knowledge
    agent.add_knowledge("Apples are red or green fruits.")
    
    # Test question
    result = agent.process_question("What are apples?")
    
    if hasattr(result, 'response'):
        response = result.response
    else:
        response = result.get('response', 'No response')
    
    print(f"Mock response: {response}")


if __name__ == "__main__":
    test_provider_directly()
    test_with_mock()