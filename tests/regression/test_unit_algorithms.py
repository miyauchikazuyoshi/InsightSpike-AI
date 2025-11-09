"""
Regression tests for existing algorithms
"""

import pytest
import numpy as np
import torch
import pytest
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

from insightspike.algorithms.graph_edit_distance import GraphEditDistance
from insightspike.algorithms.information_gain import InformationGain
from insightspike.algorithms.metrics_selector import MetricsSelector


class TestGraphEditDistanceRegression:
    """Regression tests for GED calculator"""
    
    def test_basic_ged_calculation(self):
        """Test basic GED calculation between two graphs"""
        ged_calc = GraphEditDistance()
        
        # Graph 1: 2 nodes, 1 edge
        g1 = Data(
            x=torch.randn(2, 128),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        )
        
        # Graph 2: 3 nodes, 2 edges
        g2 = Data(
            x=torch.randn(3, 128),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        )
        
        # Should calculate GED
        ged_result = ged_calc.calculate(g1, g2)
        assert hasattr(ged_result, 'ged_value')
        assert ged_result.ged_value > 0  # Graphs are different
    
    def test_identical_graphs(self):
        """Test GED for identical graphs"""
        ged_calc = GraphEditDistance()
        
        # Same graph
        g1 = Data(
            x=torch.randn(3, 64),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        )
        
        ged_result = ged_calc.calculate(g1, g1)
        assert ged_result.ged_value == 0.0  # Identical graphs
    
    def test_empty_graph_handling(self):
        """Test GED with empty graphs"""
        ged_calc = GraphEditDistance()
        
        # Empty graph
        g_empty = Data(
            x=torch.empty(0, 128),
            edge_index=torch.empty(2, 0, dtype=torch.long)
        )
        
        # Non-empty graph
        g_normal = Data(
            x=torch.randn(2, 128),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        )
        
        # Should handle empty graphs
        ged_result = ged_calc.calculate(g_empty, g_normal)
        assert hasattr(ged_result, 'ged_value')
        assert ged_result.ged_value > 0
    
    def test_error_handling(self):
        """Test error handling and fallback"""
        ged_calc = GraphEditDistance()
        
        # Create graphs that might cause dimension errors
        g1 = Data(x=torch.randn(1, 128))
        g2 = Data(x=torch.randn(1, 128))
        
        # Should not crash
        try:
            ged_result = ged_calc.calculate(g1, g2)
            assert hasattr(ged_result, 'ged_value')
        except Exception as e:
            # Should handle errors gracefully
            assert "Dimension" not in str(e) or "fallback" in str(e)


class TestInformationGainRegression:
    """Regression tests for IG calculator"""
    
    def test_basic_ig_calculation(self):
        """Test basic IG calculation"""
        ig_calc = InformationGain()
        
        # Old state: low entropy
        old_state = {
            'embeddings': np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]]),
            'labels': np.array([0, 0, 0])
        }
        
        # New state: higher entropy
        new_state = {
            'embeddings': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            'labels': np.array([0, 1, 2])
        }
        
        ig_result = ig_calc.calculate(old_state, new_state)
        assert hasattr(ig_result, 'ig_value')
        assert ig_result.ig_value >= 0  # Information gain is non-negative
    
    def test_no_change_ig(self):
        """Test IG when there's no change"""
        ig_calc = InformationGain()
        
        state = {
            'embeddings': np.random.randn(5, 64),
            'labels': np.array([0, 1, 0, 1, 0])
        }
        
        ig_result = ig_calc.calculate(state, state)
        assert ig_result.ig_value == 0.0  # No change
    
    def test_clustering_method(self):
        """Test clustering-based IG calculation"""
        ig_calc = InformationGain(method='clustering', k_clusters=3)
        
        # Random data
        old_state = {
            'embeddings': np.random.randn(10, 32)
        }
        new_state = {
            'embeddings': np.random.randn(15, 32)
        }
        
        ig_result = ig_calc.calculate(old_state, new_state)
        assert hasattr(ig_result, 'ig_value')
        assert ig_result.ig_value >= 0


class TestMetricsSelectorRegression:
    """Regression tests for MetricsSelector"""
    
    def test_default_selection(self):
        """Test default algorithm selection"""
        selector = MetricsSelector()
        
        # Should select simple algorithms by default
        info = selector.get_algorithm_info()
        assert 'ged_algorithm' in info
        assert 'ig_algorithm' in info
    
    def test_config_based_selection(self):
        """Test configuration-based selection"""
        config = {
            'algorithms': {
                'use_advanced_ged': False,
                'use_advanced_ig': False
            }
        }
        
        selector = MetricsSelector(config)
        
        # Should use simple algorithms
        assert selector.delta_ged is not None
        assert selector.delta_ig is not None
    
    def test_fallback_behavior(self):
        """Test fallback when advanced algorithms unavailable"""
        config = {
            'algorithms': {
                'use_advanced_ged': True,  # Request advanced
                'use_advanced_ig': True,
                'advanced_ged_algorithm': 'nonexistent',
                'advanced_ig_algorithm': 'nonexistent'
            }
        }
        
        selector = MetricsSelector(config)
        
        # Should fallback to simple algorithms
        assert selector.delta_ged is not None
        assert selector.delta_ig is not None
        
        # Test actual calculation
        g1 = Data(x=torch.randn(2, 64), edge_index=torch.tensor([[0], [1]], dtype=torch.long))
        g2 = Data(x=torch.randn(3, 64), edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long))
        
        ged = selector.delta_ged(g1, g2)
        assert isinstance(ged, float)
    
    def test_algorithm_info(self):
        """Test algorithm info reporting"""
        selector = MetricsSelector()
        info = selector.get_algorithm_info()
        
        assert isinstance(info, dict)
        assert 'ged_algorithm' in info
        assert 'ig_algorithm' in info
        # Check for either old or new field names
        assert 'ged_available' in info or 'advanced_available' in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
