#!/usr/bin/env python3
"""
Test 2-hop GED/IG evaluation
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from insightspike.algorithms.gedig_core import GeDIGCore
from insightspike.implementations.layers.scalable_graph_builder import ScalableGraphBuilder
from insightspike.processing.embedder import EmbeddingManager

def test_2hop_gedig():
    """Test GeDIG with 2-hop evaluation."""
    print("Testing 2-hop GED/IG evaluation...")
    
    # Create graph builder
    graph_builder = ScalableGraphBuilder({})
    embedder = EmbeddingManager()
    
    # Create test data
    texts = [
        "Energy cannot be created or destroyed",
        "Energy transforms from one form to another",
        "Conservation laws govern physical systems",
        "Physics describes fundamental principles",
        "Thermodynamics studies energy transformations"
    ]
    
    # Create embeddings
    print("Creating embeddings...")
    embeddings = []
    for text in texts:
        emb = embedder.get_embedding(text)
        embeddings.append(emb)
    embeddings = np.array(embeddings)
    
    # Build initial graph with 3 nodes
    docs1 = [
        {"id": f"doc_{i}", "text": texts[i]} 
        for i in range(3)
    ]
    embeddings1 = embeddings[:3]
    graph1 = graph_builder.build_graph(
        documents=docs1,
        embeddings=embeddings1
    )
    graph_builder.previous_graph = None  # Reset for clean build
    
    # Build expanded graph with 5 nodes
    docs2 = [
        {"id": f"doc_{i}", "text": texts[i]} 
        for i in range(5)
    ]
    embeddings2 = embeddings[:5]
    # Use incremental build to preserve node indices
    graph2 = graph_builder.build_graph(
        documents=docs2,
        embeddings=embeddings2,
        incremental=False  # Full rebuild
    )
    
    print(f"\nGraph 1: {graph1.num_nodes} nodes, {graph1.edge_index.shape[1]} edges")
    print(f"Graph 2: {graph2.num_nodes} nodes, {graph2.edge_index.shape[1]} edges")
    
    # Test 1-hop calculation
    print("\n=== 1-hop evaluation ===")
    calculator_1hop = GeDIGCore(
        enable_multihop=False,
        max_hops=1
    )
    result_1hop = calculator_1hop.calculate(graph1, graph2)
    print(f"GED: {result_1hop.ged_value:.4f}")
    print(f"IG: {result_1hop.ig_value:.4f}")
    print(f"geDIG: {result_1hop.gedig_value:.4f}")
    
    # Test 2-hop calculation
    print("\n=== 2-hop evaluation ===")
    calculator_2hop = GeDIGCore(
        enable_multihop=True,
        max_hops=2,
        decay_factor=0.5
    )
    result_2hop = calculator_2hop.calculate(graph1, graph2)
    print(f"GED: {result_2hop.ged_value:.4f}")
    print(f"IG: {result_2hop.ig_value:.4f}")
    print(f"geDIG: {result_2hop.gedig_value:.4f}")
    
    # Compare hop contributions
    if hasattr(result_2hop, 'hop_contributions'):
        print("\nHop contributions:")
        for hop, contrib in result_2hop.hop_contributions.items():
            print(f"  Hop {hop}: GED={contrib['ged']:.4f}, IG={contrib['ig']:.4f}")
    
    # Test with focal nodes (simulate new question node)
    print("\n=== 2-hop with focal node ===")
    focal_nodes = {3, 4}  # Last two added nodes
    result_focal = calculator_2hop.calculate(graph1, graph2, focal_nodes=focal_nodes)
    print(f"GED: {result_focal.ged_value:.4f}")
    print(f"IG: {result_focal.ig_value:.4f}")
    print(f"geDIG: {result_focal.gedig_value:.4f}")
    
    return result_1hop, result_2hop, result_focal

if __name__ == '__main__':
    test_2hop_gedig()