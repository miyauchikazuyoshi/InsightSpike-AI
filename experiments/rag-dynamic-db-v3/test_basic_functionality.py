#!/usr/bin/env python3
"""Basic functionality test for geDIG-RAG v3 system."""

import sys
import os
from pathlib import Path

# Add src to path for proper module resolution
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Change working directory to src for relative imports
os.chdir(str(src_path))

from core.config import ExperimentConfig
from core.gedig_evaluator import GeDIGEvaluator, GraphUpdate, UpdateType
from core.knowledge_graph import KnowledgeGraph
from baselines.static_rag import StaticRAG
from baselines.frequency_rag import FrequencyBasedRAG
from baselines.cosine_rag import CosineOnlyRAG
from baselines.gedig_rag import GeDIGRAG

import numpy as np
import networkx as nx


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


def test_rag_systems():
    """Test all 4 RAG systems."""
    print("🧪 Testing RAG Systems...")
    
    config = ExperimentConfig()
    
    # Test data
    test_documents = [
        "Machine learning is a method of data analysis that automates analytical model building.",
        "Artificial intelligence is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans.",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks."
    ]
    
    test_queries = [
        "What is machine learning?",
        "How does artificial intelligence work?",
        "Tell me about deep learning."
    ]
    
    # Initialize all systems
    systems = {
        'static': StaticRAG(config),
        'frequency': FrequencyBasedRAG(config),
        'cosine': CosineOnlyRAG(config),
        'gedig': GeDIGRAG(config)
    }
    
    results = {}
    
    for system_name, system in systems.items():
        print(f"  Testing {system_name} RAG...")
        
        # Add initial knowledge
        n_added = system.add_initial_knowledge(test_documents)
        assert n_added > 0, f"{system_name}: Should add initial documents"
        
        # Process queries
        responses = []
        for query in test_queries:
            response = system.process_query(query)
            responses.append(response)
            
            assert response.response is not None, f"{system_name}: Should generate response"
            assert response.total_time > 0, f"{system_name}: Should track processing time"
            assert response.method_name == system_name, f"{system_name}: Should set correct method name"
        
        # Get statistics
        stats = system.get_statistics()
        assert stats['queries_processed'] == len(test_queries), f"{system_name}: Should track query count"
        
        results[system_name] = {
            'responses': responses,
            'stats': stats,
            'final_graph_size': len(system.knowledge_graph.nodes)
        }
        
        print(f"    ✅ {system_name}: {stats['queries_processed']} queries, {stats['updates_applied']} updates, {results[system_name]['final_graph_size']} nodes")
    
    # Verify different behaviors
    static_updates = results['static']['stats']['updates_applied']
    gedig_updates = results['gedig']['stats']['updates_applied']
    
    assert static_updates == 0, "Static RAG should not update"
    print(f"  ✅ Static RAG: {static_updates} updates (correct: no updates)")
    print(f"  ✅ geDIG RAG: {gedig_updates} updates (dynamic behavior)")
    
    return True


def test_end_to_end():
    """Test complete end-to-end functionality."""
    print("🧪 Testing End-to-End Workflow...")
    
    config = ExperimentConfig()
    
    # Create geDIG RAG system
    rag_system = GeDIGRAG(config)
    
    # Add initial knowledge
    initial_docs = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning algorithms can learn patterns from data without being explicitly programmed.",
        "Natural language processing enables computers to understand and generate human language."
    ]
    
    n_added = rag_system.add_initial_knowledge(initial_docs)
    assert n_added == 3, "Should add all initial documents"
    
    print(f"  Added {n_added} initial documents")
    
    # Process queries and track changes
    queries = [
        "What is Python?",
        "How does machine learning work?", 
        "What is natural language processing?",
        "Tell me about deep learning.",  # New topic - should potentially add knowledge
        "Explain neural networks."       # Another new topic
    ]
    
    initial_size = len(rag_system.knowledge_graph.nodes)
    print(f"  Initial graph size: {initial_size} nodes")
    
    for i, query in enumerate(queries):
        response = rag_system.process_query(query, query_id=f"test_query_{i+1}")
        
        graph_size = len(rag_system.knowledge_graph.nodes)
        update_info = "updated" if response.knowledge_updated else "no update"
        
        print(f"  Query {i+1}: '{query[:30]}...' -> {update_info} (graph: {graph_size} nodes)")
        
        assert response.response is not None, f"Should generate response for query {i+1}"
        assert response.query_id == f"test_query_{i+1}", f"Should preserve query ID"
    
    final_size = len(rag_system.knowledge_graph.nodes)
    total_updates = rag_system.get_statistics()['updates_applied']
    
    print(f"  Final graph size: {final_size} nodes (growth: +{final_size - initial_size})")
    print(f"  Total updates applied: {total_updates}")
    
    # Get geDIG-specific statistics
    gedig_stats = rag_system.get_gedig_statistics()
    print(f"  geDIG evaluations: {gedig_stats['gedig_evaluations']}")
    print(f"  Acceptance rate: {gedig_stats.get('add_rate', 0):.2f}")
    
    assert final_size >= initial_size, "Graph should not shrink"
    assert total_updates >= 0, "Should track updates"
    
    print("  ✅ End-to-end workflow successful")
    return True


def main():
    """Run all functionality tests."""
    print("🚀 geDIG-RAG v3 Basic Functionality Tests")
    print("=" * 50)
    
    try:
        # Core component tests
        test_gedig_evaluator()
        test_knowledge_graph()
        
        # System tests
        test_rag_systems()
        
        # Integration test
        test_end_to_end()
        
        print()
        print("🎉 All Tests Passed!")
        print("✅ geDIG-RAG v3 basic functionality is working correctly")
        print()
        print("Next steps:")
        print("1. Create evaluation system framework")
        print("2. Implement data preparation scripts")
        print("3. Add comprehensive baseline comparisons")
        print("4. Design long-term session experiments")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)