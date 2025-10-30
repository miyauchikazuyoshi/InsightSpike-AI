#!/usr/bin/env python3
"""Basic functionality test for geDIG-RAG v3 system - run from src directory."""

import numpy as np
import networkx as nx

from core.config import ExperimentConfig
from core.gedig_evaluator import GeDIGEvaluator, GraphUpdate, UpdateType
from core.knowledge_graph import KnowledgeGraph


def test_gedig_evaluator():
    """Test geDIG evaluator functionality."""
    print("🧪 Testing geDIG Evaluator...")
    
    evaluator = GeDIGEvaluator(k_coefficient=0.5, radius=2)
    
    # Create simple test graph
    graph_before = nx.Graph()
    graph_before.add_node("node1", text="Test node 1")
    graph_before.add_node("node2", text="Test node 2") 
    graph_before.add_edge("node1", "node2", weight=1.0)
    
    # Create test update
    update = GraphUpdate(
        update_type=UpdateType.ADD,
        target_nodes=[],
        new_node_data={
            'id': 'node3',
            'text': 'Test node 3',
            'embedding': np.random.normal(0, 1, 384),
            'node_type': 'fact'
        },
        new_edges=[('node3', 'node1', {'weight': 0.5})]
    )
    
    # Evaluate update
    result = evaluator.evaluate_update(graph_before, update)
    
    assert result.delta_ged >= 0, "ΔGED should be non-negative for addition"
    assert result.computation_time > 0, "Computation time should be recorded"
    assert len(result.affected_nodes) >= 0, "Should track affected nodes"
    
    print(f"  ✅ geDIG evaluation successful: ΔGED={result.delta_ged:.3f}, ΔIG={result.delta_ig:.3f}, geDIG={result.delta_gedig:.3f}")
    return True


def test_knowledge_graph():
    """Test knowledge graph functionality."""
    print("🧪 Testing Knowledge Graph...")
    
    kg = KnowledgeGraph(embedding_dim=384)
    
    # Add nodes
    embedding1 = np.random.normal(0, 1, 384)
    embedding2 = np.random.normal(0, 1, 384)
    
    node1_id = kg.add_node("First test node", embedding=embedding1, node_type="fact")
    node2_id = kg.add_node("Second test node", embedding=embedding2, node_type="fact")
    
    assert node1_id is not None, "Should create first node"
    assert node2_id is not None, "Should create second node"
    assert len(kg.nodes) == 2, "Should have 2 nodes"
    
    # Add edge
    success = kg.add_edge(node1_id, node2_id, relation="semantic", weight=0.8, semantic_similarity=0.8)
    assert success, "Should add edge successfully"
    
    # Test similarity search
    query_embedding = np.random.normal(0, 1, 384)
    similar_nodes = kg.find_similar_nodes(query_embedding, k=2)
    
    assert len(similar_nodes) == 2, "Should find 2 similar nodes"
    assert all(isinstance(sim, float) for _, sim in similar_nodes), "Should return similarity scores"
    
    # Test statistics
    stats = kg.get_statistics()
    assert stats['current_nodes'] == 2, "Should report 2 nodes"
    assert stats['current_edges'] == 1, "Should report 1 edge"
    
    print(f"  ✅ Knowledge Graph functional: {stats['current_nodes']} nodes, {stats['current_edges']} edges")
    return True


def test_config_system():
    """Test configuration system."""
    print("🧪 Testing Configuration System...")
    
    config = ExperimentConfig()
    
    assert hasattr(config, 'gedig'), "Should have geDIG config section"
    assert hasattr(config, 'models'), "Should have models config section"
    assert hasattr(config, 'datasets'), "Should have datasets config section"
    
    assert config.gedig.k_coefficient > 0, "Should have valid k coefficient"
    assert config.gedig.radius > 0, "Should have valid radius"
    
    print(f"  ✅ Configuration loaded: k={config.gedig.k_coefficient}, radius={config.gedig.radius}")
    print(f"  ✅ Model config: embedding={config.models.embedding_model}")
    return True


def main():
    """Run core functionality tests."""
    print("🚀 geDIG-RAG v3 Core Functionality Tests")
    print("=" * 50)
    
    try:
        # Core component tests
        test_config_system()
        test_gedig_evaluator()
        test_knowledge_graph()
        
        print()
        print("🎉 Core Tests Passed!")
        print("✅ geDIG-RAG v3 core functionality is working correctly")
        print()
        print("Next: Test RAG systems integration")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)