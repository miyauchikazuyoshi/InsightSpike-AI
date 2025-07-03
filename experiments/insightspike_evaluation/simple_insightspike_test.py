#!/usr/bin/env python3
"""
Simple test using actual InsightSpike-AI API
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'src'))

from insightspike.algorithms.information_gain import InformationGain, EntropyMethod
from insightspike.algorithms.graph_edit_distance import GraphEditDistance, OptimizationLevel
import numpy as np

def test_information_gain():
    """Test Information Gain calculation"""
    print("Testing Information Gain...")
    
    # Create IG calculator
    ig = InformationGain(method=EntropyMethod.SHANNON)
    
    # Test data
    data_before = {
        'features': np.array([1, 0, 1, 0, 1]),
        'labels': np.array([0, 0, 1, 1, 0])
    }
    
    data_after = {
        'features': np.array([1, 1, 1, 0, 0]),
        'labels': np.array([1, 1, 1, 0, 0])
    }
    
    # Calculate IG
    result = ig.calculate(data_before, data_after)
    
    print(f"Information Gain: {result.ig_value:.3f}")
    print(f"Entropy Before: {result.entropy_before:.3f}")
    print(f"Entropy After: {result.entropy_after:.3f}")
    
    return result

def test_graph_edit_distance():
    """Test Graph Edit Distance calculation"""
    print("\nTesting Graph Edit Distance...")
    
    # Create GED calculator
    ged = GraphEditDistance(optimization_level=OptimizationLevel.STANDARD)
    
    # Create simple graphs
    import networkx as nx
    
    G1 = nx.Graph()
    G1.add_edges_from([(1, 2), (2, 3), (3, 4)])
    
    G2 = nx.Graph()
    G2.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])
    
    # Calculate GED
    result = ged.calculate(G1, G2)
    
    print(f"Graph Edit Distance: {result.ged_value}")
    # Normalize by graph size
    max_size = max(len(G1), len(G2))
    normalized_ged = result.ged_value / max_size if max_size > 0 else 0
    print(f"Normalized GED: {normalized_ged:.3f}")
    print(f"Computation time: {result.computation_time:.3f}s")
    
    return result

def test_insight_detection():
    """Test insight detection using IG and GED"""
    print("\nTesting Insight Detection...")
    
    # Get IG and GED values
    ig_result = test_information_gain()
    ged_result = test_graph_edit_distance()
    
    # Calculate geDIG score
    delta_ig = ig_result.ig_value
    max_size = max(4, 5)  # Size of test graphs
    normalized_ged = ged_result.ged_value / max_size if max_size > 0 else 0
    delta_ged = -normalized_ged  # Negative for simplification
    
    gedig_score = delta_ged * delta_ig
    
    print(f"\nInsight Detection:")
    print(f"ΔIG: {delta_ig:.3f}")
    print(f"ΔGED: {delta_ged:.3f}")
    print(f"geDIG Score: {gedig_score:.3f}")
    
    # Determine if insight
    is_insight = gedig_score < -0.1  # Threshold
    print(f"Insight Detected: {is_insight}")

def main():
    """Run all tests"""
    print("="*50)
    print("InsightSpike-AI API Test")
    print("="*50)
    
    try:
        test_insight_detection()
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()