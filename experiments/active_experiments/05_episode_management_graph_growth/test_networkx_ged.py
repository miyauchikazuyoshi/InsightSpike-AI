#!/usr/bin/env python3
"""Test NetworkX GED error in isolation"""

import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print("Testing NetworkX GED calculation...")

try:
    from insightspike.algorithms.graph_edit_distance import GraphEditDistance
    import networkx as nx
    
    # Create simple test graphs
    g1 = nx.Graph()
    g1.add_edges_from([(0, 1), (1, 2)])
    
    g2 = nx.Graph()
    g2.add_edges_from([(0, 1), (1, 2), (2, 3)])
    
    # Test GED calculation
    print("Creating GED calculator...")
    ged_calc = GraphEditDistance(optimization_level="fast")
    
    print("Calculating GED...")
    result = ged_calc.calculate(g1, g2)
    
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
    
    # Try to access ged_value
    print(f"GED value: {result.ged_value}")
    
    # Test that it's not callable
    try:
        result.ged_value()
        print("ERROR: ged_value is callable!")
    except TypeError as e:
        print(f"Good: ged_value is not callable: {e}")
    
except Exception as e:
    print(f"\nError occurred: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

print("\nTest complete.")