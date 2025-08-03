"""
Test with correct configuration
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike import MainAgent


def test_with_correct_config():
    """Test with proper configuration."""
    
    # Set API key
    os.environ["ANTHROPIC_API_KEY"] = "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"
    
    # Use a more complete configuration
    config = {
        "llm": {
            "provider": "anthropic",
            "model": "claude-3-opus-20240229",
            "temperature": 0.7,
            "max_tokens": 1000,
            "system_prompt": "You are a helpful AI assistant that answers questions based on the provided context."
        },
        "memory": {
            "max_retrieved_docs": 5,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        },
        "graph": {
            "enable_graph_search": False,
            "similarity_threshold": 0.3
        },
        "query": {
            "search_method": "sphere",
            "intuitive_radius": 0.5,
            "dimension_aware": True
        },
        "processing": {
            "enable_insight_search": False,  # Disable for simple test
            "enable_multi_stage_reasoning": False,
            "use_simple_prompt": True  # Try simple prompt mode
        },
        "layer4": {
            "provider": "anthropic",  # Explicitly set layer4 provider
            "model": "claude-3-opus-20240229"
        }
    }
    
    print("Creating agent with config...")
    agent = MainAgent(config)
    
    # Add simple knowledge
    knowledge_items = [
        "Apples are fruits that grow on trees.",
        "Apples come in many colors including red, green, and yellow.",
        "Apples are rich in fiber and vitamins."
    ]
    
    print("\nAdding knowledge...")
    for item in knowledge_items:
        agent.add_knowledge(item)
        print(f"  Added: {item}")
    
    # Test question
    question = "What colors can apples be?"
    print(f"\nAsking: {question}")
    
    result = agent.process_question(question)
    
    # Get response
    if hasattr(result, 'response'):
        response = result.response
    else:
        response = result.get('response', 'No response')
    
    print(f"Response: {response}")
    
    # Show more details
    if hasattr(result, 'graph_analysis'):
        print(f"\nGraph analysis:")
        print(f"  Spike detected: {result.graph_analysis.get('spike_detected', False)}")
        print(f"  GED: {result.graph_analysis.get('ged_value', 0):.3f}")
        print(f"  IG: {result.graph_analysis.get('ig_value', 0):.3f}")
    
    # Try another question
    question2 = "Are apples healthy?"
    print(f"\n\nAsking: {question2}")
    
    result2 = agent.process_question(question2)
    
    if hasattr(result2, 'response'):
        response2 = result2.response
    else:
        response2 = result2.get('response', 'No response')
    
    print(f"Response: {response2}")


if __name__ == "__main__":
    test_with_correct_config()