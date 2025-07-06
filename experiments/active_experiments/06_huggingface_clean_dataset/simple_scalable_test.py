#!/usr/bin/env python3
"""
Simple test of scalable graph features
======================================

Focus on demonstrating the key improvements without excessive logging.
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message=".*clustering.*")
warnings.filterwarnings("ignore", message=".*IVF-PQ.*")
warnings.filterwarnings("ignore", message=".*Exact GED.*")

from insightspike.core.layers.scalable_graph_builder import ScalableGraphBuilder
from insightspike.core.learning.scalable_graph_manager import ScalableGraphManager
from insightspike.utils.graph_importance import GraphImportanceCalculator
from insightspike.monitoring import GraphOperationMonitor
from insightspike.core.config import get_config


def test_scalable_graph_builder():
    """Test FAISS-based graph builder."""
    print("\n=== Testing Scalable Graph Builder ===")
    
    config = get_config()
    monitor = GraphOperationMonitor(enable_file_logging=False)
    builder = ScalableGraphBuilder(config, monitor)
    
    # Create test documents
    docs = []
    for i in range(100):
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        docs.append({
            "text": f"Document about topic {i % 10}",
            "embedding": embedding
        })
    
    # Build graph
    start = time.time()
    graph = builder.build_graph(docs)
    build_time = time.time() - start
    
    print(f"Built graph with {graph.num_nodes} nodes, {graph.edge_index.size(1)} edges")
    print(f"Build time: {build_time:.3f}s")
    print(f"Average edges per node: {graph.edge_index.size(1) / graph.num_nodes:.1f}")
    
    # Test neighbor search
    distances, neighbors = builder.get_neighbors(0, k=5)
    print(f"Node 0 has {len(neighbors)} neighbors")
    
    # Check monitoring
    summary = monitor.get_operation_summary()
    if "build_graph" in summary:
        print(f"Monitoring: {summary['build_graph']['count']} build operations tracked")


def test_scalable_graph_manager():
    """Test graph manager with conflict detection."""
    print("\n=== Testing Scalable Graph Manager ===")
    
    manager = ScalableGraphManager(
        similarity_threshold=0.3,
        conflict_threshold=0.85
    )
    
    # Add some nodes
    embeddings = []
    for i in range(20):
        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb)
    
    # Add conflicting episodes
    conflicts_detected = 0
    
    # Normal episodes
    for i in range(10):
        result = manager.add_episode_node(
            embeddings[i], i, 
            {"text": f"Normal document {i}"}
        )
        if result.get("conflicts"):
            conflicts_detected += len(result["conflicts"])
    
    # Add similar but conflicting content
    base_emb = embeddings[0]
    conflict_emb = base_emb + np.random.randn(384) * 0.05  # Very similar
    conflict_emb = conflict_emb / np.linalg.norm(conflict_emb)
    
    result1 = manager.add_episode_node(
        conflict_emb, 10,
        {"text": "The market will increase significantly"}
    )
    
    result2 = manager.add_episode_node(
        conflict_emb + np.random.randn(384) * 0.01, 11,
        {"text": "The market will decrease significantly"}
    )
    
    if result2.get("conflicts"):
        conflicts_detected += len(result2["conflicts"])
        print(f"Detected conflict: {result2['conflicts'][0]}")
    
    # Summary
    graph = manager.graph
    print(f"Final graph: {graph.num_nodes} nodes, {graph.edge_index.size(1)} edges")
    print(f"Total conflicts detected: {conflicts_detected}")
    
    # Test split decision
    if conflicts_detected > 0:
        should_split = manager.should_split_episode(result2.get("conflicts", []))
        print(f"Should split episode: {should_split}")


def test_graph_importance():
    """Test graph-based importance calculation."""
    print("\n=== Testing Graph Importance Calculator ===")
    
    calculator = GraphImportanceCalculator()
    
    # Create a simple graph
    import torch
    from torch_geometric.data import Data
    
    # Star graph: node 0 connected to all others
    edges = []
    for i in range(1, 10):
        edges.extend([[0, i], [i, 0]])
    
    # Add some additional connections
    edges.extend([[1, 2], [2, 1]])
    edges.extend([[3, 4], [4, 3]])
    
    graph = Data(
        x=torch.randn(10, 384),
        edge_index=torch.tensor(edges, dtype=torch.long).t()
    )
    graph.num_nodes = 10
    
    # Calculate importance for different nodes
    print("Node importance scores:")
    for node in [0, 1, 5, 9]:
        scores = calculator.calculate_importance(graph, node)
        print(f"  Node {node}: combined={scores['combined']:.3f}, "
              f"degree={scores['degree']:.3f}, pagerank={scores['pagerank']:.3f}")
    
    # Test access tracking
    for _ in range(5):
        calculator._update_access(0)
    
    scores_with_access = calculator.calculate_importance(graph, 0)
    print(f"\nNode 0 after 5 accesses: combined={scores_with_access['combined']:.3f}")
    
    # Get top important nodes
    top_nodes = calculator.get_top_k_important(graph, k=3)
    print(f"\nTop 3 important nodes: {[(n, f'{s:.3f}') for n, s in top_nodes]}")


def test_integration():
    """Test integration of all components."""
    print("\n=== Testing Component Integration ===")
    
    # Create enhanced memory manager
    from insightspike.core.layers.layer2_enhanced_scalable import L2EnhancedScalableMemory
    
    config = get_config()
    memory = L2EnhancedScalableMemory(
        dim=384,
        config=config,
        use_scalable_graph=True
    )
    
    # Add some episodes
    texts = [
        "Machine learning is transforming healthcare",
        "AI applications in medical diagnosis",
        "Deep learning for image recognition",
        "Neural networks detect cancer early",
        "Healthcare costs are increasing",  # Different topic
        "Machine learning reduces medical errors",
        "AI predicts patient outcomes",
        "Deep learning revolutionizes healthcare"  # Should integrate
    ]
    
    print(f"Adding {len(texts)} episodes...")
    for i, text in enumerate(texts):
        success = memory.store_episode(text, c_value=0.5)
        if not success:
            print(f"Failed to store: {text}")
    
    print(f"Total episodes stored: {len(memory.episodes)}")
    
    # Get graph stats
    stats = memory.get_graph_stats()
    if stats.get("graph_enabled"):
        print(f"Graph stats: {stats['nodes']} nodes, {stats['edges']} edges")
        print(f"Graph density: {stats['density']:.4f}")
        print(f"Recent conflicts: {stats['recent_conflicts']}")
    
    # Test enhanced search
    results = memory.search_episodes_with_graph("healthcare AI", k=3)
    print(f"\nSearch results for 'healthcare AI':")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['text'][:50]}... (score: {result.get('enhanced_score', result['weighted_score']):.3f})")


def main():
    """Run all tests."""
    print("=== Scalable Graph Features Test ===")
    print(f"Start time: {datetime.now()}\n")
    
    try:
        test_scalable_graph_builder()
        test_scalable_graph_manager()
        test_graph_importance()
        test_integration()
        
        print(f"\n✅ All tests completed successfully!")
        print(f"End time: {datetime.now()}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()