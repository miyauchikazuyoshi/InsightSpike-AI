#!/usr/bin/env python3
"""
Comprehensive Pipeline Test with All Config Patterns
===================================================

Tests the gedig_core integration with various configuration combinations.
"""

import sys
import traceback
from typing import Dict, List, Any

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


def create_test_graphs():
    """Create test graphs for evaluation."""
    # Simple evolution: path → cycle with added connections
    g1 = nx.path_graph(6)
    g2 = nx.cycle_graph(6)
    g2.add_edge(0, 3)  # Add cross connection
    g2.add_edge(1, 4)
    
    # Features representing semantic embeddings
    features1 = np.random.rand(6, 64)
    features2 = features1.copy()
    # Slightly modify features for new connections
    features2[0] = (features2[0] + features2[3]) / 2
    features2[1] = (features2[1] + features2[4]) / 2
    
    return g1, g2, features1, features2


def test_config_pattern(name: str, config: Dict[str, Any]):
    """Test a specific configuration pattern."""
    print(f"\n{colored(f'Testing: {name}', 'cyan', attrs=['bold'])}")
    print("-" * 50)
    
    try:
        # Create calculator and selector with config
        calculator = GeDIGCalculator(config)
        selector = MetricsSelector(config)
        
        # Get test data
        g1, g2, feat1, feat2 = create_test_graphs()
        
        # Test through calculator
        calc_result = calculator.calculate(
            g1, g2, feat1, feat2,
            focal_nodes=[0, 1]  # Focus on nodes with new connections
        )
        
        # Test through selector
        ged_value = selector.delta_ged(g1, g2)
        # Note: delta_ig in MetricsSelector uses graphs only, features are embedded in nodes
        ig_value = selector.delta_ig(g1, g2)
        
        # Display results
        print(f"Calculator results:")
        print(f"  - geDIG: {calc_result['gedig']:.4f}")
        print(f"  - GED: {calc_result['ged']:.4f}")
        print(f"  - IG: {calc_result['ig']:.4f}")
        print(f"  - Has spike: {calc_result['has_spike']}")
        
        if 'multihop_results' in calc_result:
            print(f"  - Multi-hop hops: {len(calc_result['multihop_results']['hop_details'])}")
        
        print(f"\nSelector results:")
        print(f"  - GED: {ged_value:.4f}")
        print(f"  - IG: {ig_value:.4f}")
        
        # Verify consistency
        ged_diff = abs(calc_result['ged'] - ged_value)
        ig_diff = abs(calc_result['ig'] - ig_value)
        
        if ged_diff < 0.1 and ig_diff < 0.1:
            print(colored("✓ Results consistent", "green"))
        else:
            print(colored(f"⚠ Differences: GED={ged_diff:.4f}, IG={ig_diff:.4f}", "yellow"))
        
        return True
        
    except Exception as e:
        print(colored(f"✗ Error: {str(e)}", "red"))
        traceback.print_exc()
        return False


def main():
    """Run all configuration pattern tests."""
    print(colored("=== Comprehensive Config Pattern Testing ===", "white", attrs=['bold']))
    
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
        
        # 6. Preset: experiment
        ("Preset: Experiment", ConfigPresets.get_preset("experiment")),
        
        # 7. Preset: research
        ("Preset: Research", ConfigPresets.get_preset("research")),
        
        # 8. Preset: minimal
        ("Preset: Minimal", ConfigPresets.get_preset("minimal")),
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