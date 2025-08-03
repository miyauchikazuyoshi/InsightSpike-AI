"""
Debug configuration to understand why clean provider is being used
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


def test_config_debug():
    """Debug configuration flow."""
    
    # Set API key
    os.environ["ANTHROPIC_API_KEY"] = "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"
    
    # Test configuration
    config = {
        "llm": {
            "provider": "anthropic",
            "model": "claude-3-opus-20240229",
            "temperature": 0.7,
            "max_tokens": 1000
        }
    }
    
    print("Creating agent with config:")
    print(config)
    
    # Create agent
    agent = MainAgent(config)
    
    # Check what LLM provider was created
    print(f"\nLLM provider type: {type(agent.l4_llm)}")
    print(f"LLM provider config: {agent.l4_llm.config if hasattr(agent.l4_llm, 'config') else 'No config'}")
    
    # Check if it has provider type
    if hasattr(agent.l4_llm, 'config') and hasattr(agent.l4_llm.config, 'provider'):
        print(f"Provider: {agent.l4_llm.config.provider}")
        print(f"Model: {agent.l4_llm.config.model_name if hasattr(agent.l4_llm.config, 'model_name') else 'No model'}")
    
    # Initialize and test
    if agent.initialize():
        print("\nAgent initialized successfully")
        
        # Add knowledge
        agent.add_knowledge("Test knowledge item")
        
        # Test question
        result = agent.process_question("What is test?")
        
        if hasattr(result, 'response'):
            response = result.response
        else:
            response = result.get('response', 'No response')
        
        print(f"\nResponse: {response}")
    else:
        print("\nAgent initialization failed")


if __name__ == "__main__":
    test_config_debug()