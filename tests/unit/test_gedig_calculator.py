"""
Unit tests for GeDIG Calculator with config support
"""

import pytest
import numpy as np
import networkx as nx
import torch
import pytest
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

from insightspike.algorithms.gedig_calculator import (
    GeDIGCalculator,
    calculate_gedig
)


class TestGeDIGCalculator:
    """Test suite for unified GeDIG calculator."""
    
    def test_initialization_without_config(self):
        """Test initialization without config."""
        calculator = GeDIGCalculator()
        assert calculator.config is None
        assert calculator.use_multihop is False
        assert calculator._multihop_calculator is None
    
    def test_initialization_with_basic_config(self):
        """Test initialization with basic config."""
        config = {
            'graph': {
                'ged_algorithm': 'simple',
                'ig_algorithm': 'simple'
            }
        }
        
        calculator = GeDIGCalculator(config)
        assert calculator.use_multihop is False
    
    def test_multihop_detection_from_config(self):
        """Test multi-hop detection from config."""
        # Test different config formats
        configs = [
            # Dict format
            {
                'graph': {
                    'use_multihop_gedig': True,
                    'multihop_config': {
                        'max_hops': 4,
                        'decay_factor': 0.8
                    }
                }
            },
            # Object-like format
            type('Config', (), {
                'graph': type('Graph', (), {
                    'use_multihop_gedig': True,
                    'multihop_config': {
                        'max_hops': 4,
                        'decay_factor': 0.8
                    }
                })()
            })(),
            # Direct attribute
            type('Config', (), {
                'use_multihop_gedig': True,
                'multihop_config': {
                    'max_hops': 4,
                    'decay_factor': 0.8
                }
            })()
        ]
        
        for config in configs:
            calculator = GeDIGCalculator(config)
            assert calculator.use_multihop is True
            assert calculator._multihop_calculator is not None
            assert calculator._multihop_calculator.max_hops == 4
            assert calculator._multihop_calculator.decay_factor == 0.8
    
    def test_simple_calculation(self):
        """Test simple geDIG calculation."""
        g1 = nx.path_graph(5)
        g2 = nx.cycle_graph(5)
        
        calculator = GeDIGCalculator()
        gedig = calculator.calculate_simple(g1, g2)
        
        assert isinstance(gedig, float)
    
    def test_calculation_without_multihop(self):
        """Test detailed calculation without multi-hop."""
        config = {
            'graph': {
                'use_multihop_gedig': False
            }
        }
        
        g1 = nx.Graph()
        g1.add_edges_from([(0, 1), (1, 2)])
        
        g2 = g1.copy()
        g2.add_edge(0, 2)  # Add shortcut
        
        calculator = GeDIGCalculator(config)
        result = calculator.calculate(g1, g2)
        
        assert 'gedig' in result
        assert 'ged' in result
        assert 'ig' in result
        assert 'multihop_results' not in result
        assert result['gedig'] == result['ged'] * result['ig']
    
    def test_calculation_with_multihop(self):
        """Test calculation with multi-hop enabled."""
        config = {
            'graph': {
                'use_multihop_gedig': True,
                'multihop_config': {
                    'max_hops': 2,
                    'decay_factor': 0.7
                }
            }
        }
        
        # Create hub scenario
        g_before = nx.Graph()
        g_before.add_edges_from([(0, 1), (2, 3)])
        
        g_after = g_before.copy()
        g_after.add_node(4)  # Hub
        g_after.add_edges_from([(4, 0), (4, 2)])
        
        features_before = np.random.rand(4, 10)
        features_after = np.vstack([features_before, np.random.rand(1, 10)])
        
        calculator = GeDIGCalculator(config)
        result = calculator.calculate(
            g_before, g_after,
            features_before, features_after,
            focal_nodes=[4]
        )
        
        assert 'multihop_results' in result
        assert 'total_gedig' in result['multihop_results']
        assert 'optimal_hop' in result['multihop_results']
        assert 'hop_details' in result['multihop_results']
        
        # geDIG should use multi-hop total
        assert result['gedig'] == result['multihop_results']['total_gedig']
    
    def test_multihop_without_features_warning(self):
        """Test multi-hop without features generates warning."""
        config = {
            'graph': {
                'use_multihop_gedig': True
            }
        }
        
        g1 = nx.path_graph(3)
        g2 = nx.star_graph(2)
        
        calculator = GeDIGCalculator(config)
        
        # Should not crash, but no multi-hop results
        result = calculator.calculate(g1, g2)
        assert 'multihop_results' not in result
    
    def test_pyg_data_support(self):
        """Test support for PyTorch Geometric data."""
        # Create PyG graphs
        edge_index1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        x1 = torch.randn(3, 10)
        data1 = Data(x=x1, edge_index=edge_index1)
        
        edge_index2 = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        x2 = torch.randn(3, 10)
        data2 = Data(x=x2, edge_index=edge_index2)
        
        calculator = GeDIGCalculator()
        gedig = calculator.calculate_simple(data1, data2)
        
        assert isinstance(gedig, float)
    
    def test_convenience_function(self):
        """Test convenience function."""
        g1 = nx.karate_club_graph()
        g2 = nx.erdos_renyi_graph(34, 0.1)
        
        # Simple call
        gedig_simple = calculate_gedig(g1, g2)
        assert isinstance(gedig_simple, float)
        
        # With config
        config = {'graph': {'ged_algorithm': 'simple'}}
        gedig_config = calculate_gedig(g1, g2, config=config)
        assert isinstance(gedig_config, float)
        
        # With multi-hop args
        features1 = np.random.rand(34, 20)
        features2 = np.random.rand(34, 20)
        
        result = calculate_gedig(
            g1, g2,
            features_before=features1,
            features_after=features2
        )
        assert isinstance(result, dict)
        assert 'gedig' in result
    
    def test_config_parameter_propagation(self):
        """Test that config parameters properly propagate."""
        config = {
            'graph': {
                'use_multihop_gedig': True,
                'multihop_config': {
                    'max_hops': 5,
                    'decay_factor': 0.6,
                    'adaptive_hops': False,
                    'min_improvement': 0.2,
                    'ged_weight': 0.7,
                    'ig_weight': 0.3
                }
            }
        }
        
        calculator = GeDIGCalculator(config)
        
        # Verify all parameters
        assert calculator._multihop_calculator.max_hops == 5
        assert calculator._multihop_calculator.decay_factor == 0.6
        assert calculator._multihop_calculator.adaptive_hops is False
        assert calculator._multihop_calculator.min_improvement == 0.2
        assert calculator._multihop_calculator.ged_weight == 0.7
        assert calculator._multihop_calculator.ig_weight == 0.3
    
    def test_default_multihop_config(self):
        """Test default multi-hop configuration."""
        config = {
            'graph': {
                'use_multihop_gedig': True
                # No multihop_config provided
            }
        }
        
        calculator = GeDIGCalculator(config)
        
        # Should use defaults
        assert calculator._multihop_calculator.max_hops == 3
        assert calculator._multihop_calculator.decay_factor == 0.7
        assert calculator._multihop_calculator.adaptive_hops is True
    
    def test_edge_cases(self):
        """Test edge cases."""
        calculator = GeDIGCalculator()
        
        # Empty graphs
        g_empty = nx.Graph()
        g_single = nx.Graph()
        g_single.add_node(0)
        
        result = calculator.calculate_simple(g_empty, g_single)
        assert isinstance(result, float)
        
        # Same graph
        g = nx.complete_graph(5)
        result = calculator.calculate_simple(g, g)
        assert isinstance(result, float)
        assert result == 0.0  # No change


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
