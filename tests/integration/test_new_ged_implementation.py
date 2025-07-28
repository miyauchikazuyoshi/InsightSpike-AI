#!/usr/bin/env python3
"""
Test New GED Implementation with Feature Flag
============================================
"""

import numpy as np
import networkx as nx
from typing import Dict, Any


def test_with_feature_flag():
    """Test new GED implementation using feature flag."""
    print("=== Testing New GED Implementation ===\n")
    
    # Create test graphs
    G1 = nx.Graph()
    G1.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])  # Square
    
    G2 = nx.Graph()
    G2.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])  # Square
    G2.add_edges_from([(4, 0), (4, 1), (4, 2), (4, 3)])  # Add hub
    
    # Test with old implementation
    print("1. Testing OLD implementation:")
    from insightspike.algorithms.metrics_selector import MetricsSelector
    
    old_config = type('Config', (), {'use_new_ged_implementation': False})()
    old_selector = MetricsSelector(old_config)
    old_ged = old_selector.delta_ged(G1, G2)
    print(f"   Old GED: {old_ged}")
    
    # Test with new implementation
    print("\n2. Testing NEW implementation:")
    new_config = type('Config', (), {'use_new_ged_implementation': True})()
    new_selector = MetricsSelector(new_config)
    
    # First, create the analyzer to test directly
    from insightspike.algorithms.graph_structure_analyzer import GraphStructureAnalyzer
    analyzer = GraphStructureAnalyzer()
    result = analyzer.analyze_structural_change(G1, G2)
    
    print(f"   Raw GED: {result['ged']}")
    print(f"   Structural Improvement: {result['structural_improvement']:.3f}")
    print(f"   Efficiency Change: {result['efficiency_change']:.3f}")
    print(f"   Hub Formation: {result['hub_formation']:.3f}")
    
    # Test through selector (with backward compatibility)
    new_ged = new_selector.delta_ged(G1, G2)
    print(f"\n   GED through selector: {new_ged}")
    print(f"   (Should be negative because structure improved)")
    
    # Test spike detection with both
    print("\n3. Spike Detection Comparison:")
    print(f"   Old: GED={old_ged:.3f} < -0.5? {old_ged < -0.5}")
    print(f"   New: GED={new_ged:.3f} < -0.5? {new_ged < -0.5}")
    
    # Test integrated metrics
    print("\n4. Testing Integrated GEDIG Metrics:")
    from insightspike.metrics.improved_gedig_metrics import calculate_gedig_metrics
    
    # Create dummy embeddings
    embeddings1 = np.random.randn(4, 384)
    embeddings2 = np.random.randn(5, 384)
    
    metrics = calculate_gedig_metrics(G1, G2, embeddings1, embeddings2)
    print(f"   GED: {metrics.ged}")
    print(f"   IG: {metrics.ig:.3f}")
    print(f"   Structural Improvement: {metrics.structural_improvement:.3f}")
    print(f"   Knowledge Coherence: {metrics.knowledge_coherence:.3f}")
    print(f"   Insight Score: {metrics.insight_score:.3f}")
    print(f"   Spike Detected: {metrics.spike_detected}")
    print(f"   Spike Intensity: {metrics.spike_intensity:.3f}")


def test_edge_cases():
    """Test edge cases for new implementation."""
    print("\n\n=== Testing Edge Cases ===\n")
    
    from insightspike.algorithms.graph_structure_analyzer import GraphStructureAnalyzer
    analyzer = GraphStructureAnalyzer()
    
    # Empty to single node
    G1 = nx.Graph()
    G2 = nx.Graph()
    G2.add_node(0)
    
    result = analyzer.analyze_structural_change(G1, G2)
    print(f"Empty → Single: GED={result['ged']}, Improvement={result['structural_improvement']:.3f}")
    
    # Disconnected to connected
    G1 = nx.Graph()
    G1.add_edges_from([(0, 1), (2, 3)])
    
    G2 = nx.Graph()
    G2.add_edges_from([(0, 1), (1, 2), (2, 3)])
    
    result = analyzer.analyze_structural_change(G1, G2)
    print(f"Disconnected → Connected: GED={result['ged']}, Improvement={result['structural_improvement']:.3f}")


def benchmark_performance():
    """Benchmark old vs new implementation."""
    print("\n\n=== Performance Benchmark ===\n")
    
    import time
    
    # Create larger graphs
    G1 = nx.erdos_renyi_graph(20, 0.3)
    G2 = nx.erdos_renyi_graph(20, 0.4)
    
    # Old implementation
    from insightspike.algorithms.metrics_selector import MetricsSelector
    old_selector = MetricsSelector()
    
    start = time.time()
    for _ in range(10):
        old_ged = old_selector.delta_ged(G1, G2)
    old_time = time.time() - start
    
    # New implementation
    new_config = type('Config', (), {'use_new_ged_implementation': True})()
    new_selector = MetricsSelector(new_config)
    
    start = time.time()
    for _ in range(10):
        new_ged = new_selector.delta_ged(G1, G2)
    new_time = time.time() - start
    
    print(f"Old implementation: {old_time:.3f}s for 10 runs")
    print(f"New implementation: {new_time:.3f}s for 10 runs")
    print(f"Speed ratio: {old_time/new_time:.2f}x")


if __name__ == "__main__":
    test_with_feature_flag()
    test_edge_cases()
    benchmark_performance()