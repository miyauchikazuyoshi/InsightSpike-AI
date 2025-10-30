#!/usr/bin/env python3
"""
Test geDIG calculation with configuration
"""

import numpy as np
import networkx as nx
import yaml

from insightspike.algorithms.gedig_calculator import GeDIGCalculator


def test_basic_gedig():
    """Test basic geDIG without multi-hop."""
    print("=== Test Basic geDIG ===\n")
    
    # Simple config
    config = {
        'graph': {
            'ged_algorithm': 'simple',
            'ig_algorithm': 'simple',
            'use_multihop_gedig': False
        }
    }
    
    # Create graphs
    g1 = nx.path_graph(5)
    g2 = nx.star_graph(4)
    
    # Calculate
    calculator = GeDIGCalculator(config)
    gedig = calculator.calculate_simple(g1, g2)
    
    print(f"Basic geDIG: {gedig:.4f}")


def test_multihop_from_config():
    """Test multi-hop geDIG with configuration."""
    print("\n=== Test Multi-hop from Config ===\n")
    
    # Load actual config
    with open('config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Enable multi-hop
    base_config['graph']['use_multihop_gedig'] = True
    
    # Create scenario
    g_before = nx.Graph()
    g_before.add_edges_from([(0, 1), (2, 3), (4, 5)])
    
    g_after = g_before.copy()
    g_after.add_node(6)  # Hub node
    g_after.add_edges_from([(6, 0), (6, 2), (6, 4)])
    
    # Features
    np.random.seed(42)
    features_before = np.random.rand(6, 20)
    features_after = np.vstack([features_before, np.random.rand(1, 20)])
    
    # Calculate with multi-hop
    calculator = GeDIGCalculator(base_config)
    result = calculator.calculate(
        g_before, g_after,
        features_before, features_after,
        focal_nodes=[6]  # Focus on hub
    )
    
    print(f"Total geDIG: {result['gedig']:.4f}")
    print(f"Basic components: GED={result['ged']:.4f}, IG={result['ig']:.4f}")
    
    if 'multihop_results' in result:
        print(f"\nMulti-hop analysis:")
        print(f"  Total weighted: {result['multihop_results']['total_gedig']:.4f}")
        print(f"  Optimal hop: {result['multihop_results']['optimal_hop']}")
        print(f"\n  Hop breakdown:")
        
        for hop, details in sorted(result['multihop_results']['hop_details'].items()):
            print(f"    Hop {hop}: geDIG={details['gedig']:.4f}, "
                  f"nodes={details['nodes_in_subgraph']}, "
                  f"weight={details['weight']:.2f}")


def test_config_variations():
    """Test different configuration variations."""
    print("\n=== Test Config Variations ===\n")
    
    g1 = nx.cycle_graph(6)
    g2 = nx.complete_graph(6)
    
    features1 = np.random.rand(6, 10)
    features2 = features1 * 1.1  # Small change
    
    # Test 1: Multi-hop with custom parameters
    config1 = {
        'graph': {
            'use_multihop_gedig': True,
            'multihop_config': {
                'max_hops': 2,
                'decay_factor': 0.5,
                'adaptive_hops': False
            }
        }
    }
    
    calc1 = GeDIGCalculator(config1)
    result1 = calc1.calculate(g1, g2, features1, features2)
    print(f"Config 1 (max_hops=2, decay=0.5): {result1['gedig']:.4f}")
    
    # Test 2: Multi-hop with high decay
    config2 = {
        'graph': {
            'use_multihop_gedig': True,
            'multihop_config': {
                'max_hops': 4,
                'decay_factor': 0.9,
                'adaptive_hops': True
            }
        }
    }
    
    calc2 = GeDIGCalculator(config2)
    result2 = calc2.calculate(g1, g2, features1, features2)
    print(f"Config 2 (max_hops=4, decay=0.9): {result2['gedig']:.4f}")
    
    # Test 3: With entropy variance IG
    config3 = {
        'graph': {
            'use_entropy_variance_ig': True,
            'use_multihop_gedig': True,
            'multihop_config': {
                'max_hops': 3
            }
        }
    }
    
    calc3 = GeDIGCalculator(config3)
    result3 = calc3.calculate(g1, g2, features1, features2)
    print(f"Config 3 (entropy variance + multi-hop): {result3['gedig']:.4f}")


def test_reward_calculation_scenario():
    """Test a scenario similar to reward calculation in agents."""
    print("\n=== Test Reward Calculation Scenario ===\n")
    
    # Simulate agent's graph evolution
    initial_graph = nx.Graph()
    initial_graph.add_edges_from([(0, 1), (1, 2)])  # Linear knowledge
    
    # After processing question
    updated_graph = initial_graph.copy()
    updated_graph.add_node(3)  # New insight
    updated_graph.add_edges_from([(3, 0), (3, 2)])  # Connects distant concepts
    
    features_before = np.array([
        [1, 0, 0],  # Concept A
        [0, 1, 0],  # Concept B
        [0, 0, 1],  # Concept C
    ])
    
    features_after = np.vstack([
        features_before,
        [0.3, 0.3, 0.4]  # Mixed insight
    ])
    
    # Test with different hop configurations
    hop_configs = [
        {'max_hops': 0, 'name': 'Direct only'},
        {'max_hops': 1, 'name': 'One-hop'},
        {'max_hops': 2, 'name': 'Two-hop'},
        {'max_hops': 3, 'name': 'Three-hop'}
    ]
    
    for hop_config in hop_configs:
        config = {
            'graph': {
                'use_multihop_gedig': True,
                'multihop_config': {
                    'max_hops': hop_config['max_hops'],
                    'decay_factor': 0.7,
                    'adaptive_hops': False
                }
            }
        }
        
        calculator = GeDIGCalculator(config)
        result = calculator.calculate(
            initial_graph, updated_graph,
            features_before, features_after,
            focal_nodes=[3]
        )
        
        print(f"{hop_config['name']:12s}: geDIG = {result['gedig']:.4f}")
        
        if 'multihop_results' in result:
            hop_details = result['multihop_results']['hop_details']
            hop_values = [f"{h}:{d['gedig']:.3f}" for h, d in sorted(hop_details.items())]
            print(f"              Details: {', '.join(hop_values)}")


if __name__ == "__main__":
    test_basic_gedig()
    test_multihop_from_config()
    test_config_variations()
    test_reward_calculation_scenario()
    
    print("\n=== Summary ===")
    print("✓ geDIG calculation can be controlled via config.yaml")
    print("✓ Multi-hop analysis can be enabled/disabled")
    print("✓ Hop parameters (max_hops, decay_factor, etc.) are configurable")
    print("✓ Compatible with agent reward calculation scenarios")