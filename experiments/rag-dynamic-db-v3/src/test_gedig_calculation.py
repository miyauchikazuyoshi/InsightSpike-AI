#!/usr/bin/env python3
"""Test geDIG calculation with actual values to understand why no updates."""

from run_improved_gedig import ImprovedGeDIGSystem, ExperimentConfig, create_high_quality_knowledge_base
import numpy as np

def test_gedig_calc():
    """Test geDIG calculation step by step."""
    print("🔍 Testing geDIG Calculation")
    print("=" * 60)
    
    # Setup with the most lenient configuration
    config = ExperimentConfig()
    params = {
        'k': 0.01,  # Very low k
        'node_weight': 1.0,  # Very high node weight
        'edge_weight': 0.5,  # High edge weight  
        'novelty_weight': 0.1,  # Low novelty weight
        'connectivity_weight': 0.05,
        'threshold_base': -1.0,  # Very lenient
        'threshold_novelty_high': -2.0,
        'threshold_novelty_low': -0.5,
        'edge_similarity_threshold': 0.1,  # Very low threshold
        'adaptive_k': False
    }
    
    system = ImprovedGeDIGSystem("gedig", config, params)
    
    # Add initial knowledge
    knowledge_base = create_high_quality_knowledge_base()
    system.add_initial_knowledge(knowledge_base[:3])  # Just a few items
    
    print(f"\nInitial setup:")
    print(f"  Nodes: {len(system.nx_graph.nodes)}")
    print(f"  Edges: {len(system.nx_graph.edges)}")
    print(f"  Density: {system.initial_density:.3f}")
    
    # Test query
    query = "What is quantum computing?"
    query_embedding = system.embedder.encode(query)[0]
    
    # Find similar nodes
    similar_nodes = system.knowledge_graph.find_similar_nodes(
        query_embedding, k=5, min_similarity=0.0
    )
    
    print(f"\nQuery: {query}")
    print(f"Similar nodes found: {len(similar_nodes)}")
    if similar_nodes:
        print(f"Max similarity: {similar_nodes[0][1]:.3f}")
    
    # Manually calculate what geDIG should be
    response = "Quantum computing uses quantum mechanics principles"
    
    # Simulate update
    nodes_added = 1
    
    # Count potential edges
    edges_added = 0
    for node_id, sim in similar_nodes[:5]:
        if sim > params['edge_similarity_threshold']:
            edges_added += 1
            print(f"  Would add edge to {node_id[:20]}... with sim={sim:.3f}")
    
    # Calculate components
    ged = nodes_added * params['node_weight'] + edges_added * params['edge_weight']
    
    max_sim = similar_nodes[0][1] if similar_nodes else 0.0
    novelty = 1.0 - max_sim
    connectivity = edges_added * params['connectivity_weight']
    ig = novelty * params['novelty_weight'] + connectivity
    
    gedig_score = ged - params['k'] * ig
    
    # Determine threshold
    if novelty > 0.7:
        threshold = params['threshold_novelty_high']
    elif novelty > 0.5:
        threshold = params['threshold_base'] - 0.1
    elif novelty > 0.3:
        threshold = params['threshold_base']
    else:
        threshold = params['threshold_novelty_low']
    
    print(f"\n📊 Manual Calculation:")
    print(f"  Nodes added: {nodes_added}")
    print(f"  Edges added: {edges_added}")
    print(f"  GED = {nodes_added} × {params['node_weight']} + {edges_added} × {params['edge_weight']} = {ged:.3f}")
    print(f"  Novelty = 1 - {max_sim:.3f} = {novelty:.3f}")
    print(f"  IG = {novelty:.3f} × {params['novelty_weight']} + {connectivity:.3f} = {ig:.3f}")
    print(f"  geDIG = {ged:.3f} - {params['k']} × {ig:.3f} = {gedig_score:.3f}")
    print(f"  Threshold = {threshold:.3f}")
    print(f"  Decision: {'UPDATE' if gedig_score > threshold else 'SKIP'}")
    
    # Now run actual evaluation
    should_update, metadata = system._evaluate_with_gedig(query, response, similar_nodes)
    
    print(f"\n📊 Actual Evaluation:")
    print(f"  geDIG score: {metadata['gedig_score']:.3f}")
    print(f"  Threshold: {metadata['threshold_used']:.3f}")
    print(f"  Decision: {'UPDATE' if should_update else 'SKIP'}")
    
    # Try even more extreme parameters
    print("\n" + "=" * 60)
    print("🔥 Testing EXTREME Configuration")
    
    extreme_params = {
        'k': 0.0,  # No IG penalty at all!
        'node_weight': 2.0,
        'edge_weight': 1.0,
        'novelty_weight': 0.0,  # Ignore novelty completely
        'connectivity_weight': 0.0,
        'threshold_base': -0.001,  # Almost always accept
        'threshold_novelty_high': -0.001,
        'threshold_novelty_low': -0.001,
        'edge_similarity_threshold': 0.01,  # Accept any similarity
        'adaptive_k': False
    }
    
    system2 = ImprovedGeDIGSystem("gedig", config, extreme_params)
    system2.add_initial_knowledge(knowledge_base[:3])
    
    similar_nodes2 = system2.knowledge_graph.find_similar_nodes(
        query_embedding, k=5, min_similarity=0.0
    )
    
    should_update2, metadata2 = system2._evaluate_with_gedig(query, response, similar_nodes2)
    
    print(f"  geDIG score: {metadata2['gedig_score']:.3f}")
    print(f"  Threshold: {metadata2['threshold_used']:.3f}")
    print(f"  Decision: {'UPDATE' if should_update2 else 'SKIP'}")
    
    if not should_update2:
        print("\n⚠️ Even extreme configuration doesn't update!")
        print("  This suggests an issue with the evaluation logic itself.")

if __name__ == "__main__":
    test_gedig_calc()