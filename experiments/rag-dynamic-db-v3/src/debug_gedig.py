#!/usr/bin/env python3
"""Debug geDIG evaluation to understand why updates are not happening."""

from run_experiment_improved import (
    ImprovedEmbedder,
    create_high_quality_knowledge_base,
    ExperimentConfig
)
from run_parameter_sweep import ParameterizedGeDIGSystem
import numpy as np

def debug_gedig_evaluation():
    """Debug a single geDIG evaluation."""
    print("🔍 Debugging geDIG Evaluation")
    print("=" * 60)
    
    # Setup
    config = ExperimentConfig()
    
    # Test with Ultra Aggressive parameters
    params = {
        'k': 0.05,
        'node_weight': 0.3,
        'edge_weight': 0.2,
        'novelty_weight': 0.3,
        'connectivity_weight': 0.2,
        'threshold_base': -0.05,
        'threshold_novelty_high': -0.3,
        'threshold_novelty_low': 0.0
    }
    
    system = ParameterizedGeDIGSystem("gedig", config, params)
    
    # Add initial knowledge
    knowledge_base = create_high_quality_knowledge_base()
    n_added = system.add_initial_knowledge(knowledge_base[:3])  # Just a few for debugging
    print(f"\nInitial knowledge: {n_added} items")
    print(f"Initial edges: {len(system.knowledge_graph.edges)}")
    
    # Test queries
    test_queries = [
        ("What is transfer learning?", "technical"),  # Novel
        ("How does Python's GIL work?", "technical"),  # Somewhat related
        ("Explain quantum computing", "conceptual"),  # Very novel
    ]
    
    print("\n" + "-" * 60)
    for i, (query, depth) in enumerate(test_queries):
        print(f"\n🔍 Query {i+1}: {query}")
        
        # Get embedding and find similar nodes
        query_embedding = system.embedder.encode(query)[0]
        similar_nodes = system.knowledge_graph.find_similar_nodes(
            query_embedding, k=5, min_similarity=0.0
        )
        
        max_similarity = similar_nodes[0][1] if similar_nodes else 0.0
        print(f"   Max similarity: {max_similarity:.3f}")
        
        # Generate response
        response = f"Test response for: {query}"
        
        # Manually evaluate with geDIG
        should_update, metadata = system._evaluate_with_gedig(query, response, similar_nodes)
        
        print(f"\n   📊 geDIG Evaluation:")
        print(f"      GED: {metadata.get('ged', 0):.3f}")
        print(f"      IG: {metadata.get('ig', 0):.3f}")
        print(f"      geDIG Score: {metadata.get('gedig_score', 0):.3f}")
        print(f"      Novelty: {metadata.get('novelty', 0):.3f}")
        print(f"      Threshold: {metadata.get('threshold_used', 0):.3f}")
        print(f"      Decision: {'UPDATE' if should_update else 'SKIP'}")
        
        # Show calculation details
        print(f"\n   📐 Calculation Details:")
        print(f"      nodes_added × {params['node_weight']} = 1 × {params['node_weight']} = {1 * params['node_weight']:.3f}")
        print(f"      edges_added × {params['edge_weight']} = {metadata.get('edges_added', 0)} × {params['edge_weight']} = {metadata.get('edges_added', 0) * params['edge_weight']:.3f}")
        print(f"      GED total = {metadata.get('ged', 0):.3f}")
        print(f"      ")
        print(f"      novelty × {params['novelty_weight']} = {metadata.get('novelty', 0):.3f} × {params['novelty_weight']} = {metadata.get('novelty', 0) * params['novelty_weight']:.3f}")
        print(f"      connectivity × {params['connectivity_weight']} = {metadata.get('edges_added', 0)} × {params['connectivity_weight']} = {metadata.get('edges_added', 0) * params['connectivity_weight']:.3f}")
        print(f"      IG total = {metadata.get('ig', 0):.3f}")
        print(f"      ")
        print(f"      geDIG = GED - k × IG = {metadata.get('ged', 0):.3f} - {params['k']} × {metadata.get('ig', 0):.3f} = {metadata.get('gedig_score', 0):.3f}")
        print(f"      threshold = {metadata.get('threshold_used', 0):.3f}")
        print(f"      {metadata.get('gedig_score', 0):.3f} > {metadata.get('threshold_used', 0):.3f}? {should_update}")

def test_different_scenarios():
    """Test different scenarios to find working parameters."""
    print("\n\n🧪 Testing Different Scenarios")
    print("=" * 60)
    
    scenarios = [
        {"name": "Extreme", "k": 0.01, "node_weight": 0.5, "edge_weight": 0.3, "threshold": -0.5},
        {"name": "NoIG", "k": 0.0, "node_weight": 0.3, "edge_weight": 0.2, "threshold": 0.0},
        {"name": "HighGED", "k": 0.1, "node_weight": 1.0, "edge_weight": 0.5, "threshold": 0.0},
    ]
    
    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")
        
        # Simulate values
        nodes_added = 1
        edges_added = 2
        novelty = 0.8
        
        ged = nodes_added * scenario['node_weight'] + edges_added * scenario['edge_weight']
        ig = novelty * 0.5 + edges_added * 0.2
        gedig = ged - scenario['k'] * ig
        
        print(f"   GED = {ged:.3f}")
        print(f"   IG = {ig:.3f}") 
        print(f"   geDIG = {gedig:.3f}")
        print(f"   Threshold = {scenario['threshold']:.3f}")
        print(f"   Result: {'✅ UPDATE' if gedig > scenario['threshold'] else '❌ SKIP'}")

if __name__ == "__main__":
    debug_gedig_evaluation()
    test_different_scenarios()