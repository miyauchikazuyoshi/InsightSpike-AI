#!/usr/bin/env python3
"""
Quick test for normalized geDIG calculation
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from insightspike.algorithms.gedig_core_normalize import GeDIGNormalizedCalculator
import networkx as nx
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sign_consistency():
    """Test that the sign of the reward is consistent."""
    
    # Create test config
    config = {
        # Base gedig_core config
        "w1": 0.5,
        "kT": 1.0,
        "node_cost": 1.0,
        "edge_cost": 0.5,
        
        # Normalization config
        "normalization": {
            "enabled": True,
            "mode": "conservation",
            "size_normalization": {"beta": 0.5},
            "reward": {"lambda_ig": 1.0, "mu_ged": 0.5},
            "spike_detection": {"mode": "threshold", "threshold": 0.0},
            "z_transform": {"use_running_stats": True, "window_size": 10}
        }
    }
    
    calculator = GeDIGNormalizedCalculator(config)
    
    # Test 1: Adding nodes (graph grows)
    print("\n=== Test 1: Graph Growth ===")
    g1 = nx.Graph()
    g1.add_nodes_from([0, 1])
    g1.add_edge(0, 1)
    
    g2 = nx.Graph()
    g2.add_nodes_from([0, 1, 2])
    g2.add_edges_from([(0, 1), (1, 2), (0, 2)])
    
    result1 = calculator.calculate(g1, g2)
    print(f"Growth: raw_GED={result1['ged']:.3f}, IG={result1['ig']:.3f}")
    print(f"Graph size: before={result1['statistics']['graph_size_before']}, after={result1['statistics']['graph_size_after']}")
    print(f"Normalized: GED_norm={result1['normalized_metrics']['ged_normalized']:.3f}")
    print(f"Conservation sum: {result1['normalized_metrics']['conservation_sum']:.3f}")
    print(f"IG Z-score: {result1['normalized_metrics']['ig_z_score']:.3f}")
    print(f"Reward R={result1['gedig']:.3f}, Spike={result1['has_spike']}")
    
    # Test 2: Removing nodes (graph shrinks - insight!)
    print("\n=== Test 2: Graph Simplification ===")
    result2 = calculator.calculate(g2, g1)
    print(f"Simplify: GED={result2['ged']:.3f}, IG={result2['ig']:.3f}")
    print(f"Normalized: GED_norm={result2['normalized_metrics']['ged_normalized']:.3f}")
    print(f"Conservation sum: {result2['normalized_metrics']['conservation_sum']:.3f}")
    print(f"IG Z-score: {result2['normalized_metrics']['ig_z_score']:.3f}")
    print(f"Reward R={result2['gedig']:.3f}, Spike={result2['has_spike']}")
    
    # Test 3: Multiple changes to build statistics
    print("\n=== Test 3: Building Statistics ===")
    for i in range(5):
        g3 = nx.Graph()
        g3.add_nodes_from(range(i+3))
        g3.add_edges_from([(j, j+1) for j in range(i+2)])
        
        result = calculator.calculate(g2, g3)
        print(f"Step {i+1}: R={result['gedig']:.3f}, IG_z={result['normalized_metrics']['ig_z_score']:.3f}")
    
    # Summary
    print("\n=== Summary ===")
    print(f"IG stats: mean={calculator.ig_mean:.3f}, std={calculator.ig_std:.3f}")
    print(f"Config: {calculator.get_config_summary()}")

if __name__ == "__main__":
    test_sign_consistency()