#!/usr/bin/env python3
"""
Check current graph state in memory
===================================
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.layers.layer2_enhanced_scalable import L2EnhancedScalableMemory
from insightspike.core.config import get_config


def check_graph_state():
    """Check the current graph state in memory and on disk."""
    
    print("=== Checking Graph State ===\n")
    
    # Initialize agent
    agent = MainAgent()
    
    # Replace with enhanced memory
    config = get_config()
    agent.l2_memory = L2EnhancedScalableMemory(
        dim=config.embedding.dimension,
        config=config,
        use_scalable_graph=True
    )
    
    agent.initialize()
    
    # Check memory state
    print("1. Memory State:")
    memory_stats = agent.l2_memory.get_memory_stats()
    print(f"   Episodes: {memory_stats.get('total_episodes', 0)}")
    print(f"   Active: {memory_stats.get('active_episodes', 0)}")
    
    # Check graph state
    print("\n2. Graph State (in memory):")
    if agent.l3_graph and hasattr(agent.l3_graph, 'previous_graph'):
        if agent.l3_graph.previous_graph:
            graph = agent.l3_graph.previous_graph
            print(f"   Nodes: {graph.num_nodes}")
            print(f"   Edges: {graph.edge_index.size(1) if hasattr(graph, 'edge_index') else 0}")
            print(f"   Graph object: {type(graph)}")
        else:
            print("   No graph in memory (previous_graph is None)")
    else:
        print("   L3GraphReasoner not available or no previous_graph attribute")
    
    # Check scalable graph
    print("\n3. Scalable Graph State:")
    if hasattr(agent.l2_memory, 'get_graph_stats'):
        graph_stats = agent.l2_memory.get_graph_stats()
        print(f"   Graph enabled: {graph_stats.get('graph_enabled', False)}")
        print(f"   Nodes: {graph_stats.get('nodes', 0)}")
        print(f"   Edges: {graph_stats.get('edges', 0)}")
        print(f"   Recent conflicts: {graph_stats.get('recent_conflicts', 0)}")
    
    # Check disk files
    print("\n4. Disk Files:")
    data_dir = Path("data")
    
    # Check graph file
    graph_file = data_dir / "graph_pyg.pt"
    if graph_file.exists():
        print(f"   graph_pyg.pt: EXISTS ({graph_file.stat().st_size} bytes)")
        print(f"   Last modified: {graph_file.stat().st_mtime}")
    else:
        print("   graph_pyg.pt: NOT FOUND")
    
    # Check episodes file
    episodes_file = data_dir / "episodes.json"
    if episodes_file.exists():
        print(f"   episodes.json: EXISTS ({episodes_file.stat().st_size} bytes)")
        with open(episodes_file, 'r') as f:
            episodes = json.load(f)
            print(f"   Episodes count: {len(episodes)}")
    else:
        print("   episodes.json: NOT FOUND")
    
    # Check FAISS index
    index_file = data_dir / "index.faiss"
    if index_file.exists():
        print(f"   index.faiss: EXISTS ({index_file.stat().st_size} bytes)")
    else:
        print("   index.faiss: NOT FOUND")
    
    # Try to load and check graph
    print("\n5. Loading Graph from Disk:")
    try:
        if agent.load_state():
            print("   ✓ Successfully loaded state")
            
            # Check loaded graph
            if agent.l3_graph and agent.l3_graph.previous_graph:
                graph = agent.l3_graph.previous_graph
                print(f"   Loaded graph: {graph.num_nodes} nodes, {graph.edge_index.size(1) if hasattr(graph, 'edge_index') else 0} edges")
            else:
                print("   No graph loaded")
                
            # Check loaded memory
            memory_stats = agent.l2_memory.get_memory_stats()
            print(f"   Loaded episodes: {memory_stats.get('total_episodes', 0)}")
        else:
            print("   ✗ Failed to load state")
    except Exception as e:
        print(f"   ✗ Error loading state: {e}")
    
    # Get memory-graph state
    print("\n6. Combined Memory-Graph State:")
    state = agent.get_memory_graph_state()
    print(f"   {json.dumps(state, indent=2)}")
    
    return agent


if __name__ == "__main__":
    agent = check_graph_state()
    
    print("\n=== Testing Graph Update ===")
    
    # Add a test episode
    test_text = "This is a test episode to see graph updates."
    result = agent.add_episode_with_graph_update(test_text)
    
    print(f"\nAdded episode result:")
    print(f"   Success: {result.get('success', False)}")
    print(f"   Episode index: {result.get('episode_idx', -1)}")
    
    if result.get('graph_analysis'):
        analysis = result['graph_analysis']
        print(f"   Graph nodes: {analysis.get('metrics', {}).get('graph_nodes', 0)}")
        print(f"   Graph edges: {analysis.get('metrics', {}).get('graph_edges', 0)}")
    
    # Save state
    print("\n=== Saving State ===")
    if agent.save_state():
        print("✓ State saved successfully")
    else:
        print("✗ Failed to save state")