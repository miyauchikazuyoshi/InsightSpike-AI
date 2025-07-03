#!/usr/bin/env python3
"""
Test different methods to make graph grow
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.layers.layer3_graph_reasoner import L3GraphReasoner

def test_cumulative_documents():
    """Test if passing all documents makes graph grow"""
    print("=== Testing Cumulative Document Method ===\n")
    
    # Initialize agent
    agent = MainAgent()
    agent.initialize()
    
    # Track all documents
    all_documents = []
    
    # Add multiple episodes
    texts = [
        "Artificial intelligence is transforming healthcare.",
        "Machine learning algorithms detect patterns in data.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computer understanding.",
        "Computer vision allows machines to interpret images."
    ]
    
    for i, text in enumerate(texts):
        print(f"\nAdding episode {i+1}: {text[:50]}...")
        
        # Create document
        result = agent.add_episode_with_graph_update(text)
        
        if result['success']:
            # Add to cumulative documents
            doc = {
                "text": text,
                "embedding": result['vector'],
                "c_value": result['c_value']
            }
            all_documents.append(doc)
            
            # Now analyze ALL documents cumulatively
            if agent.l3_graph:
                # Pass all documents so far
                graph_analysis = agent.l3_graph.analyze_documents(all_documents)
                
                # Check graph state
                if agent.l3_graph.previous_graph:
                    num_nodes = agent.l3_graph.previous_graph.num_nodes
                    print(f"  Graph nodes after update: {num_nodes}")
                    print(f"  Total documents in graph: {len(all_documents)}")
    
    # Save state
    agent.save_state()
    
    # Check final graph
    graph_state = agent.get_memory_graph_state()
    print(f"\nFinal graph state: {graph_state}")
    
    return agent

def test_direct_graph_manipulation():
    """Test direct graph building"""
    print("\n=== Testing Direct Graph Building ===\n")
    
    # Initialize L3GraphReasoner directly
    reasoner = L3GraphReasoner()
    
    # Create multiple documents
    documents = []
    for i in range(10):
        doc = {
            "text": f"Document {i}: This is test content about topic {i}",
            "embedding": [0.1 * i] * 384,  # Simple test embedding
            "c_value": 0.5
        }
        documents.append(doc)
    
    # Build graph with all documents
    graph = reasoner.graph_builder.build_graph(documents)
    print(f"Built graph with {graph.num_nodes} nodes from {len(documents)} documents")
    
    # Save the graph
    reasoner.save_graph(graph)
    print("Graph saved to disk")
    
    # Load and verify
    loaded_graph = reasoner.load_graph()
    if loaded_graph:
        print(f"Loaded graph has {loaded_graph.num_nodes} nodes")
    
    return reasoner

if __name__ == "__main__":
    # Test cumulative method
    agent = test_cumulative_documents()
    
    # Test direct method
    reasoner = test_direct_graph_manipulation()
    
    print("\n=== Conclusion ===")
    print("Graph growth requires passing ALL documents to analyze_documents()")
    print("The graph is rebuilt each time, not incrementally updated")