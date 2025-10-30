"""
Test Spectral GED Enhancement
============================

Tests for the spectral evaluation feature in geDIG.
"""

import numpy as np
import networkx as nx
import pytest

from insightspike.algorithms.gedig_core import GeDIGCore


class TestSpectralGED:
    """Test spectral GED evaluation functionality."""
    
    def test_spectral_disabled_by_default(self):
        """Ensure spectral evaluation is disabled by default."""
        calculator = GeDIGCore()
        assert not calculator.enable_spectral
        
    def test_spectral_can_be_enabled(self):
        """Test enabling spectral evaluation."""
        calculator = GeDIGCore(enable_spectral=True, spectral_weight=0.4)
        assert calculator.enable_spectral
        assert calculator.spectral_weight == 0.4
        
    def test_backward_compatibility(self):
        """Ensure results are identical when spectral is disabled."""
        # Create test graphs
        g1 = nx.Graph()
        g1.add_edges_from([(0, 1), (1, 2), (2, 3)])
        
        g2 = nx.Graph()
        g2.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        
        # Use fixed features to avoid randomness
        np.random.seed(42)
        features1 = np.random.randn(4, 64)
        features2 = np.random.randn(5, 64)
        
        # Calculate with spectral disabled
        calc_off = GeDIGCore(enable_spectral=False)
        result_off = calc_off.calculate(g1, g2, features1, features2)
        
        # Calculate with spectral disabled (different instance)
        calc_off2 = GeDIGCore()
        result_off2 = calc_off2.calculate(g1, g2, features1, features2)
        
        # Results should be identical
        assert result_off.gedig_value == result_off2.gedig_value
        assert result_off.ged_value == result_off2.ged_value
        assert result_off.ig_value == result_off2.ig_value
        
    def test_spectral_affects_results(self):
        """Test that enabling spectral changes the results."""
        # Create test graphs with different structures
        g1 = nx.Graph()
        g1.add_edges_from([(0, 1), (1, 2), (2, 3)])  # Path graph
        
        g2 = nx.Graph() 
        g2.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)])  # More connected
        
        # Use fixed features
        np.random.seed(42)
        features1 = np.random.randn(4, 64)
        features2 = np.random.randn(4, 64)  # Same number of nodes
        
        # Calculate with spectral disabled
        calc_off = GeDIGCore(enable_spectral=False)
        result_off = calc_off.calculate(g1, g2, features1, features2)
        
        # Calculate with spectral enabled
        calc_on = GeDIGCore(enable_spectral=True, spectral_weight=0.5)
        result_on = calc_on.calculate(g1, g2, features1, features2)
        
        # Results should be different
        assert result_off.structural_improvement != result_on.structural_improvement
        # The overall gedig value should be different
        assert result_off.gedig_value != result_on.gedig_value
        
    def test_spectral_score_calculation(self):
        """Test the spectral score calculation directly."""
        calculator = GeDIGCore()
        
        # Empty graph
        g_empty = nx.Graph()
        assert calculator._calculate_spectral_score(g_empty) == 0.0
        
        # Single node
        g_single = nx.Graph()
        g_single.add_node(0)
        assert calculator._calculate_spectral_score(g_single) == 0.0
        
        # Regular graph (path)
        g_path = nx.path_graph(5)
        score_path = calculator._calculate_spectral_score(g_path)
        assert score_path > 0
        
        # Star graph (one central node connected to all others)
        g_star = nx.star_graph(4)  # 5 nodes total
        score_star = calculator._calculate_spectral_score(g_star)
        assert score_star > 0
        
        # Different structures should have different scores
        assert score_star != score_path
        
        # Test that we can compute spectral scores for various graphs
        graphs = [
            nx.cycle_graph(6),
            nx.complete_graph(4),
            nx.ladder_graph(3),
            nx.wheel_graph(5)
        ]
        
        scores = [calculator._calculate_spectral_score(g) for g in graphs]
        # All should be positive
        assert all(s > 0 for s in scores)
        # And they should be different
        assert len(set(scores)) == len(scores)
        
    def test_config_based_spectral(self):
        """Test spectral configuration via config dict."""
        from insightspike.algorithms.gedig_core import calculate_gedig
        
        # Create test graphs
        g1 = nx.Graph()
        g1.add_edges_from([(0, 1), (1, 2)])
        
        g2 = nx.Graph()
        g2.add_edges_from([(0, 1), (1, 2), (2, 3)])
        
        # Test with spectral disabled
        config_off = {
            'metrics': {
                'spectral_evaluation': {
                    'enabled': False,
                    'weight': 0.3
                }
            }
        }
        result_off = calculate_gedig(g1, g2, config=config_off)
        
        # Test with spectral enabled
        config_on = {
            'metrics': {
                'spectral_evaluation': {
                    'enabled': True,
                    'weight': 0.3
                }
            }
        }
        result_on = calculate_gedig(g1, g2, config=config_on)
        
        # Results should be different
        assert result_off != result_on


if __name__ == "__main__":
    pytest.main([__file__, "-v"])