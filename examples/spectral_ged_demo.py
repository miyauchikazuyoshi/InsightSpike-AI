#!/usr/bin/env python3
"""
Spectral GED Demonstration
=========================

This script demonstrates how spectral GED evaluation affects insight detection.
"""

import networkx as nx
import numpy as np
from insightspike.algorithms.gedig_core import GeDIGCore
from insightspike.algorithms.linkset_adapter import build_linkset_info

def demonstrate_spectral_ged():
    """Demonstrate spectral GED with different graph structures."""
    
    print("Spectral GED Demonstration")
    print("=" * 50)
    
    # Create different graph structures
    # 1. Disconnected graph (before)
    g1 = nx.Graph()
    g1.add_edges_from([(0, 1), (2, 3), (4, 5)])  # Three disconnected pairs
    
    # 2. Hub-connected graph (after)
    g2 = nx.Graph()
    g2.add_edges_from([(0, 1), (2, 3), (4, 5),  # Original edges
                       (6, 0), (6, 2), (6, 4)])  # Hub node 6 connects all
    
    # Generate some features
    np.random.seed(42)
    features1 = np.random.randn(6, 64)
    features2 = np.random.randn(7, 64)
    
    print("\nScenario: Adding a hub node to connect disconnected components")
    print(f"Before: {g1.number_of_nodes()} nodes, {g1.number_of_edges()} edges")
    print(f"After: {g2.number_of_nodes()} nodes, {g2.number_of_edges()} edges")
    
    # Calculate without spectral
    print("\n1. Without Spectral Evaluation:")
    calc_no_spectral = GeDIGCore(enable_spectral=False)
    ls = build_linkset_info(s_link=[{"index": 1, "similarity": 1.0}], candidate_pool=[], decision={"index": 1, "similarity": 1.0}, query_vector=[1.0], base_mode="link")
    result_no_spectral = calc_no_spectral.calculate(g1, g2, features1, features2, linkset_info=ls)
    
    print(f"   GED: {result_no_spectral.ged_value:.4f}")
    print(f"   IG: {result_no_spectral.ig_value:.4f}")
    print(f"   geDIG: {result_no_spectral.gedig_value:.4f}")
    print(f"   Structural Improvement: {result_no_spectral.structural_improvement:.4f}")
    
    # Calculate with spectral
    print("\n2. With Spectral Evaluation (weight=0.5):")
    calc_spectral = GeDIGCore(enable_spectral=True, spectral_weight=0.5)
    result_spectral = calc_spectral.calculate(g1, g2, features1, features2, linkset_info=ls)
    
    print(f"   GED: {result_spectral.ged_value:.4f}")
    print(f"   IG: {result_spectral.ig_value:.4f}")
    print(f"   geDIG: {result_spectral.gedig_value:.4f}")
    print(f"   Structural Improvement: {result_spectral.structural_improvement:.4f}")
    
    # Show spectral scores
    print("\n3. Spectral Analysis:")
    spectral_before = calc_spectral._calculate_spectral_score(g1)
    spectral_after = calc_spectral._calculate_spectral_score(g2)
    
    print(f"   Spectral score before: {spectral_before:.4f}")
    print(f"   Spectral score after: {spectral_after:.4f}")
    print(f"   Change: {spectral_after - spectral_before:.4f}")
    
    # Interpretation
    print("\n4. Interpretation:")
    if result_spectral.structural_improvement < result_no_spectral.structural_improvement:
        print("   ✓ Spectral evaluation detected structural improvement!")
        print("   The hub node created a more organized/regular structure.")
    else:
        print("   The structure became less organized.")
    
    # Another example: Random edges vs structured edges
    print("\n" + "=" * 50)
    print("Scenario 2: Random edges vs Structured edges")
    
    # Create a path graph
    g3 = nx.path_graph(8)
    
    # Add random edges
    g4_random = g3.copy()
    g4_random.add_edges_from([(0, 7), (2, 6), (1, 5)])  # Random connections
    
    # Add structured edges (complete the cycle)
    g4_structured = g3.copy()
    g4_structured.add_edge(7, 0)  # Complete the cycle
    
    features3 = np.random.randn(8, 64)
    
    print(f"\nBase: Path graph with {g3.number_of_edges()} edges")
    print(f"Random: Added 3 random edges → {g4_random.number_of_edges()} edges")
    print(f"Structured: Added 1 edge to complete cycle → {g4_structured.number_of_edges()} edges")
    
    # Compare results
    result_random = calc_spectral.calculate(g3, g4_random, features3, features3, linkset_info=ls)
    result_structured = calc_spectral.calculate(g3, g4_structured, features3, features3, linkset_info=ls)
    
    print(f"\nRandom edges - geDIG: {result_random.gedig_value:.4f}")
    print(f"Structured edge - geDIG: {result_structured.gedig_value:.4f}")
    
    if result_structured.gedig_value < result_random.gedig_value:
        print("\n✓ Spectral evaluation correctly identified that completing")
        print("  the cycle creates better structure than random edges!")


if __name__ == "__main__":
    demonstrate_spectral_ged()
