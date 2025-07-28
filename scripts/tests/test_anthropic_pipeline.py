#!/usr/bin/env python3
"""
Test pipeline with Anthropic API
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from insightspike.implementations.agents.main_agent import MainAgent

logging.basicConfig(level=logging.WARNING)

def main():
    print("\n=== ANTHROPIC API TEST ===\n")
    
    # Set API key
    os.environ["ANTHROPIC_API_KEY"] = "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"
    
    # Config with Anthropic and message passing
    config = {
        "llm": {
            "provider": "anthropic",
            "model": "claude-3-sonnet-20240229",
            "api_key": os.environ.get("ANTHROPIC_API_KEY"),
            "temperature": 0.7,
            "max_tokens": 500
        },
        "graph": {
            "enable_message_passing": True,
            "message_passing": {"alpha": 0.3, "iterations": 2}
        }
    }
    
    print("Creating agent with Anthropic API...")
    agent = MainAgent(config)
    print("✓ Agent created")
    
    # Add knowledge
    print("\nAdding knowledge...")
    knowledge = [
        "Apples are fruits that grow on trees.",
        "Apple Inc. is a technology company founded by Steve Jobs.",
        "Neural networks are inspired by biological neurons.",
        "Deep learning uses multiple layers of neural networks."
    ]
    
    for k in knowledge:
        agent.add_knowledge(k)
    print(f"✓ Added {len(knowledge)} knowledge items")
    
    # Test question
    print("\nTesting question...")
    question = "What is an apple?"
    
    try:
        result = agent.process_question(question)
        
        # Extract fields
        response = getattr(result, 'response', 'No response')
        spike = getattr(result, 'spike_detected', False)
        
        print(f"\nQuestion: {question}")
        print(f"Response: {response}")
        print(f"Spike detected: {spike}")
        
        # Check if message passing worked
        graph_analysis = getattr(result, 'graph_analysis', {})
        if graph_analysis:
            print(f"\nGraph analysis available: {bool(graph_analysis.get('graph'))}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()