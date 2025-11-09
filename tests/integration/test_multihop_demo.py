#!/usr/bin/env python3
"""
Demo of Multi-hop geDIG with Entropy Variance IG (legacy).

Cloud-safe guard: this demo depends on legacy modules merged into GeDIGCore
and on matplotlib animations; skip in constrained environments.
"""

import pytest
pytest.skip(
    "Legacy multihop demo relies on merged modules and heavy viz; skip in cloud runs.",
    allow_module_level=True,
)

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from insightspike.algorithms.multihop_gedig import MultiHopGeDIG
from insightspike.algorithms.entropy_variance_ig import EntropyVarianceIG
from insightspike.algorithms.normalized_ged import NormalizedGED


def create_insight_scenario():
    """Create a scenario demonstrating multi-hop insight propagation."""
    # Before: Disconnected knowledge clusters
    g_before = nx.Graph()
    # Math cluster
    g_before.add_edges_from([(0, 1), (1, 2)])  # 0: addition, 1: multiplication, 2: division
    # Logic cluster  
    g_before.add_edges_from([(3, 4)])  # 3: if-then, 4: boolean
    # Geometry cluster
    g_before.add_edges_from([(5, 6), (6, 7)])  # 5: triangle, 6: angle, 7: parallel
    
    # After: Central insight connects concepts
    g_after = g_before.copy()
    g_after.add_node(8)  # 8: algebraic structure (insight)
    # Connect to one node from each cluster
    g_after.add_edges_from([(8, 1), (8, 3), (8, 6)])
    
    # Features representing concept embeddings
    np.random.seed(42)
    features_before = np.zeros((8, 20))
    # Math cluster features
    features_before[0:3, 0:7] = np.random.rand(3, 7) + 1
    # Logic cluster features  
    features_before[3:5, 7:14] = np.random.rand(2, 7) + 1
    # Geometry cluster features
    features_before[5:8, 14:20] = np.random.rand(3, 6) + 1
    
    # After: insight node has mixed features
    features_after = np.vstack([
        features_before,
        np.concatenate([
            np.mean(features_before[0:3, 0:7], axis=0),
            np.mean(features_before[3:5, 7:14], axis=0),
            np.mean(features_before[5:8, 14:20], axis=0)
        ]).reshape(1, -1)
    ])
    
    return g_before, g_after, features_before, features_after


def analyze_multihop_gedig():
    """Analyze geDIG at different hop levels."""
    g_before, g_after, feat_before, feat_after = create_insight_scenario()
    
    print("=== Multi-hop geDIG Analysis ===\n")
    
    # Initialize calculator
    calculator = MultiHopGeDIG(
        max_hops=3,
        decay_factor=0.7,
        adaptive_hops=False  # Show all hops
    )
    
    # Calculate
    result = calculator.calculate(
        g_before, g_after,
        feat_before, feat_after,
        focal_nodes=[8]  # Focus on insight node
    )
    
    print(f"Total weighted geDIG: {result.total_gedig:.4f}")
    print(f"Optimal hop level: {result.optimal_hop}\n")
    
    # Detailed hop analysis
    print("Hop-by-hop breakdown:")
    print("-" * 70)
    print("Hop | Nodes | Edges | GED     | IG      | geDIG   | Weight | Weighted")
    print("-" * 70)
    
    for hop in sorted(result.hop_results.keys()):
        r = result.hop_results[hop]
        print(f"{hop:3d} | {r['nodes_in_subgraph']:5d} | {r['edges_in_subgraph']:5d} | "
              f"{r['ged']:7.4f} | {r['ig']:7.4f} | {r['gedig']:7.4f} | "
              f"{r['weight']:6.3f} | {r['weighted_gedig']:8.4f}")
    
    return result


def compare_ig_methods():
    """Compare different IG calculation methods."""
    g_before, g_after, feat_before, feat_after = create_insight_scenario()
    
    print("\n\n=== Comparing IG Methods ===\n")
    
    # 1. Entropy Variance IG
    ev_calc = EntropyVarianceIG()
    ev_result = ev_calc.calculate(g_after, feat_before, feat_after)
    
    print(f"1. Entropy Variance IG: {ev_result.ig_value:.4f}")
    print(f"   - Variance before: {ev_result.variance_before:.4f}")
    print(f"   - Variance after: {ev_result.variance_after:.4f}")
    print(f"   - Mean entropy change: {ev_result.mean_entropy_before:.4f} -> {ev_result.mean_entropy_after:.4f}")
    
    # 2. Try with LocalInformationGainV2 if available
    try:
        from insightspike.algorithms.local_information_gain_v2 import LocalInformationGainV2
        local_calc = LocalInformationGainV2()
        local_result = local_calc.calculate(g_before, g_after)
        print(f"\n2. Local Information Gain V2: {local_result.total_ig:.4f}")
        print(f"   - Global IG: {local_result.global_ig:.4f}")
        print(f"   - Homogenization: {local_result.homogenization:.4f}")
        print(f"   - Tension reduction: {local_result.tension_reduction:.4f}")
    except ImportError:
        print("\n2. Local Information Gain V2: Not available")


def visualize_multihop_subgraphs():
    """Visualize the subgraphs at each hop level."""
    g_before, g_after, feat_before, feat_after = create_insight_scenario()
    
    calculator = MultiHopGeDIG(max_hops=3, adaptive_hops=False)
    result = calculator.calculate(
        g_before, g_after,
        feat_before, feat_after,
        focal_nodes=[8]
    )
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for hop, ax in enumerate(axes):
        if hop in result.hop_results:
            # Extract k-hop subgraph
            subgraph, _ = calculator._extract_k_hop_subgraph(g_after, [8], hop)
            
            # Draw
            pos = nx.spring_layout(subgraph, seed=42)
            nx.draw(subgraph, pos, ax=ax, with_labels=True, 
                   node_color='lightblue', node_size=500,
                   font_size=10, font_weight='bold')
            
            # Highlight focal node
            if 0 in subgraph:  # Node 8 is relabeled to 0 in subgraph
                nx.draw_networkx_nodes(subgraph, pos, nodelist=[0], 
                                     node_color='red', node_size=600, ax=ax)
            
            ax.set_title(f"Hop {hop}\nNodes: {result.hop_results[hop]['nodes_in_subgraph']}\n"
                        f"geDIG: {result.hop_results[hop]['gedig']:.3f}")
        else:
            ax.text(0.5, 0.5, "Not computed", ha='center', va='center')
            ax.set_title(f"Hop {hop}")
        
        ax.axis('off')
    
    plt.suptitle("Multi-hop Subgraphs", fontsize=14)
    plt.tight_layout()
    plt.savefig("multihop_subgraphs.png", dpi=150, bbox_inches='tight')
    print("\n\nVisualization saved as 'multihop_subgraphs.png'")


def test_mathematical_reasoning():
    """Test on a mathematical reasoning scenario."""
    print("\n\n=== Mathematical Reasoning Test ===\n")
    
    # Create a math problem graph
    g_before = nx.Graph()
    # Basic arithmetic
    g_before.add_edges_from([(0, 1)])  # 0: 2+3, 1: =5
    
    # After reasoning: connect to deeper concepts
    g_after = g_before.copy()
    # Add reasoning steps
    g_after.add_node(2)  # Commutative property
    g_after.add_node(3)  # Number line
    g_after.add_node(4)  # Group theory
    
    # Direct connections
    g_after.add_edges_from([(1, 2), (1, 3)])
    # Indirect connections
    g_after.add_edges_from([(2, 4)])
    
    # Simple features
    feat_before = np.array([[1, 0], [0, 1]])
    feat_after = np.array([
        [1, 0],      # 2+3
        [0, 1],      # =5
        [0.5, 0.5],  # Commutative
        [0.7, 0.3],  # Number line
        [0.3, 0.7]   # Group theory
    ])
    
    calculator = MultiHopGeDIG(max_hops=3, decay_factor=0.8)
    result = calculator.calculate(
        g_before, g_after,
        feat_before, feat_after,
        focal_nodes=[1]  # Focus on the answer node
    )
    
    print("Problem: 2 + 3 = ?")
    print("\nInsight depth analysis:")
    for hop in sorted(result.hop_results.keys()):
        r = result.hop_results[hop]
        interpretations = {
            0: "Direct answer (5)",
            1: "Basic properties (commutativity, number line)",
            2: "Abstract concepts (group theory)",
            3: "Full mathematical context"
        }
        print(f"  Hop {hop}: {interpretations.get(hop, 'Extended context')}")
        print(f"    - geDIG: {r['gedig']:.4f} (weighted: {r['weighted_gedig']:.4f})")
    
    print(f"\nTotal insight score: {result.total_gedig:.4f}")
    print(f"Deepest meaningful hop: {result.optimal_hop}")


if __name__ == "__main__":
    # Run demonstrations
    result = analyze_multihop_gedig()
    compare_ig_methods()
    visualize_multihop_subgraphs()
    test_mathematical_reasoning()
    
    print("\n\n=== Summary ===")
    print("Multi-hop geDIG enables:")
    print("1. Detection of insights at different abstraction levels")
    print("2. Quantification of reasoning depth")
    print("3. Identification of optimal explanation granularity")
    print("\nEntropy variance IG provides:")
    print("1. Theoretically grounded information integration measure")
    print("2. Scale-invariant metric")
    print("3. Interpretable variance reduction")
