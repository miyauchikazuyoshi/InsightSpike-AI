"""
Simple test to verify LLM is working correctly
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike import MainAgent


def test_simple_qa():
    """Test basic question answering."""
    print("Testing InsightSpike with simple Q&A...")
    
    # Set API key
    os.environ["ANTHROPIC_API_KEY"] = "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"
    
    # Simple config
    config = {
        "llm": {
            "provider": "anthropic",
            "model": "claude-3-opus-20240229",
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "memory": {
            "max_retrieved_docs": 5
        },
        "processing": {
            "enable_insight_search": False
        }
    }
    
    # Create agent
    agent = MainAgent(config)
    
    # Add some knowledge
    knowledge_items = [
        "Apples are fruits that grow on trees.",
        "Apples come in many colors including red, green, and yellow.",
        "Apples are rich in fiber and vitamins.",
        "Apple trees bloom in spring with white or pink flowers.",
        "The saying 'An apple a day keeps the doctor away' promotes eating apples for health."
    ]
    
    print("\nAdding knowledge items...")
    for item in knowledge_items:
        agent.add_knowledge(item)
        print(f"  Added: {item}")
    
    # Test questions
    questions = [
        "What are apples?",
        "What colors can apples be?",
        "Are apples healthy?"
    ]
    
    print("\nTesting questions...")
    for question in questions:
        print(f"\nQ: {question}")
        
        try:
            result = agent.process_question(question)
            
            # Get response
            if hasattr(result, 'response'):
                response = result.response
            else:
                response = result.get('response', 'No response')
            
            print(f"A: {response}")
            
            # Check if spike detected
            if hasattr(result, 'has_spike'):
                print(f"Spike detected: {result.has_spike}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_simple_qa()