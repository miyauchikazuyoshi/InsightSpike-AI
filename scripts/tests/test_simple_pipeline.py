#!/usr/bin/env python3
"""
Simple pipeline test to verify basic functionality.
"""

import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from insightspike.implementations.agents.main_agent import MainAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("\n=== SIMPLE PIPELINE TEST ===\n")
    
    # Set API key
    os.environ["ANTHROPIC_API_KEY"] = "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"
    
    # Simple config
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
            "message_passing": {
                "alpha": 0.3,
                "iterations": 2
            }
        }
    }
    
    print("Creating agent...")
    try:
        agent = MainAgent(config)
        print("✓ Agent created")
    except Exception as e:
        print(f"✗ Failed to create agent: {e}")
        return
    
    # Add knowledge
    print("\nAdding knowledge...")
    agent.add_knowledge("The sun is a star at the center of our solar system.")
    agent.add_knowledge("Photosynthesis is the process by which plants convert sunlight into energy.")
    agent.add_knowledge("Solar panels convert sunlight into electricity.")
    print("✓ Knowledge added")
    
    # Test multiple questions
    questions = [
        "How does the sun relate to energy production?",
        "What's the connection between photosynthesis and solar panels?",
        "How do plants and solar panels both use sunlight?"
    ]
    
    print("\nTesting questions...")
    for i, question in enumerate(questions, 1):
        print(f"\n--- Question {i} ---")
        print(f"Q: {question}")
        
        try:
            result = agent.process_question(question)
            
            # Extract response
            if hasattr(result, 'response'):
                response = result.response
                has_spike = getattr(result, 'spike_detected', False)
                graph_analysis = getattr(result, 'graph_analysis', {})
            else:
                response = result.get('response', 'No response')
                has_spike = result.get('spike_detected', False)
                graph_analysis = result.get('graph_analysis', {})
            
            print(f"Response: {response[:150]}..." if len(response) > 150 else f"Response: {response}")
            print(f"Spike detected: {has_spike}")
            
            # Check for message passing effects
            if graph_analysis:
                graph = graph_analysis.get('graph')
                if graph and hasattr(graph, 'edge_info') and graph.edge_info:
                    new_edges = sum(1 for e in graph.edge_info if e.get('type') == 'new')
                    total_edges = len(graph.edge_info)
                    print(f"Edges: {total_edges} total ({new_edges} new)")
                
        except Exception as e:
            print(f"✗ Failed to process question: {e}")
            import traceback
            traceback.print_exc()
    
    # Check memory state
    print("\n--- Memory Summary ---")
    stats = agent.l2_memory.get_memory_stats()
    print(f"Total episodes: {stats.get('total_episodes', 0)}")
    print(f"Vector dimension: {stats.get('vector_dimension', 0)}")

if __name__ == "__main__":
    main()