#!/usr/bin/env python3
"""
Comprehensive Pipeline Test with All Config Patterns (Fixed Version)
===================================================================

Tests the gedig_core integration with various configuration combinations.
This version properly embeds features into graph nodes.
"""

import sys
import traceback
from typing import Dict, List, Any, Tuple

import networkx as nx
import numpy as np

from insightspike.algorithms.gedig_calculator import GeDIGCalculator
from insightspike.algorithms.metrics_selector import MetricsSelector
from insightspike.config.presets import ConfigPresets


# Simple color functions
def colored(text, color, attrs=None):
    """Simple colored text for terminal output."""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'white': '\033[97m',
    }
    reset = '\033[0m'
    bold = '\033[1m' if attrs and 'bold' in attrs else ''
    return f"{bold}{colors.get(color, '')}{text}{reset}"


def create_test_graphs_with_features() -> Tuple[nx.Graph, nx.Graph, np.ndarray, np.ndarray]:
    """Create test graphs with embedded features."""
    # Create base graphs
    g1 = nx.path_graph(6)
    g2 = nx.cycle_graph(6)
    g2.add_edge(0, 3)  # Add cross connections
    g2.add_edge(1, 4)
    
    # Create features
    np.random.seed(42)
    features1 = np.random.rand(6, 64)
    features2 = features1.copy()
    
    # Modify features for nodes with new connections
    features2[0] = (features2[0] + features2[3]) / 2
    features2[1] = (features2[1] + features2[4]) / 2
    
    # Embed features into graph nodes
    for i in range(6):
        g1.nodes[i]['feature'] = features1[i]
        g1.nodes[i]['vec'] = features1[i]  # Some code expects 'vec'
        g2.nodes[i]['feature'] = features2[i]
        g2.nodes[i]['vec'] = features2[i]
    
    return g1, g2, features1, features2


def test_config_pattern(name: str, config: Dict[str, Any]):
    """Test a specific configuration pattern."""
    print(f"\n{colored(f'Testing: {name}', 'cyan', attrs=['bold'])}")
    print("-" * 50)
    
    try:
        # Create calculator and selector with config
        calculator = GeDIGCalculator(config)
        selector = MetricsSelector(config)
        
        # Get test data with embedded features
        g1, g2, feat1, feat2 = create_test_graphs_with_features()
        
        # Test through calculator (supports external features)
        calc_result = calculator.calculate(
            g1, g2, feat1, feat2,
            focal_nodes=[0, 1]  # Focus on nodes with new connections
        )
        
        # Test through selector (uses embedded features)
        ged_value = selector.delta_ged(g1, g2)
        ig_value = selector.delta_ig(g1, g2)
        
        # Also test direct calculation with features for comparison
        from insightspike.algorithms.gedig_core import GeDIGCore
        direct_calc = GeDIGCore(
            enable_multihop=config.get('graph', {}).get('use_multihop_gedig', False),
            max_hops=config.get('graph', {}).get('multihop_config', {}).get('max_hops', 3)
        )
        direct_result = direct_calc.calculate(g1, g2, feat1, feat2)
        
        # Display results
        print(f"Calculator results:")
        print(f"  - geDIG: {calc_result['gedig']:.4f}")
        print(f"  - GED: {calc_result['ged']:.4f}")
        print(f"  - IG: {calc_result['ig']:.4f}")
        print(f"  - Has spike: {calc_result['has_spike']}")
        
        if 'multihop_results' in calc_result:
            print(f"  - Multi-hop hops: {len(calc_result['multihop_results']['hop_details'])}")
        
        print(f"\nSelector results (embedded features):")
        print(f"  - GED: {ged_value:.4f}")
        print(f"  - IG: {ig_value:.4f}")
        
        print(f"\nDirect core results (with features):")
        print(f"  - geDIG: {direct_result.gedig_value:.4f}")
        print(f"  - GED: {direct_result.ged_value:.4f}")
        print(f"  - IG: {direct_result.ig_value:.4f}")
        
        # Check consistency between calculator and direct core
        core_diff = abs(calc_result['gedig'] - direct_result.gedig_value)
        
        if core_diff < 0.0001:
            print(colored("✓ Calculator and core consistent", "green"))
        else:
            print(colored(f"⚠ Core difference: {core_diff:.4f}", "yellow"))
        
        return True
        
    except Exception as e:
        print(colored(f"✗ Error: {str(e)}", "red"))
        traceback.print_exc()
        return False


def main():
    """Run all configuration pattern tests."""
    print(colored("=== Comprehensive Config Pattern Testing (Fixed) ===", "white", attrs=['bold']))
    
    # Define test configurations
    test_configs = [
        # 1. Basic configuration (all features disabled)
        ("Basic (all disabled)", {
            'graph': {
                'ged_algorithm': 'simple',
                'ig_algorithm': 'simple',
                'use_entropy_variance_ig': False,
                'use_normalized_ged': False,
                'use_multihop_gedig': False,
            }
        }),
        
        # 2. Entropy variance IG only
        ("Entropy Variance IG", {
            'graph': {
                'ged_algorithm': 'simple',
                'ig_algorithm': 'simple',
                'use_entropy_variance_ig': True,
                'use_normalized_ged': False,
                'use_multihop_gedig': False,
            }
        }),
        
        # 3. Normalized GED only
        ("Normalized GED", {
            'graph': {
                'ged_algorithm': 'simple',
                'ig_algorithm': 'simple',
                'use_entropy_variance_ig': False,
                'use_normalized_ged': True,
                'use_multihop_gedig': False,
            }
        }),
        
        # 4. Multi-hop geDIG only
        ("Multi-hop geDIG", {
            'graph': {
                'ged_algorithm': 'simple',
                'ig_algorithm': 'simple',
                'use_entropy_variance_ig': False,
                'use_normalized_ged': False,
                'use_multihop_gedig': True,
                'multihop_config': {
                    'max_hops': 2,
                    'decay_factor': 0.7,
                    'adaptive_hops': True,
                }
            }
        }),
        
        # 5. All features enabled
        ("All Features", {
            'graph': {
                'ged_algorithm': 'simple',
                'ig_algorithm': 'simple',
                'use_entropy_variance_ig': True,
                'use_normalized_ged': True,
                'use_multihop_gedig': True,
                'multihop_config': {
                    'max_hops': 3,
                    'decay_factor': 0.8,
                    'adaptive_hops': True,
                }
            }
        }),
    ]
    
    # Run tests
    results = []
    for name, config in test_configs:
        success = test_config_pattern(name, config)
        results.append((name, success))
    
    # Summary
    print(f"\n{colored('=== Test Summary ===', 'white', attrs=['bold'])}")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = colored("PASS", "green") if success else colored("FAIL", "red")
        print(f"{name:.<40} {status}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        print(colored("\n✨ All tests passed! ✨", "green", attrs=['bold']))
        return 0
    else:
        print(colored(f"\n⚠ {total - passed} tests failed", "red", attrs=['bold']))
        return 1


if __name__ == "__main__":
    sys.exit(main())