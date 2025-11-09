"""
Unit tests for NormalizedGED (legacy path).

Skip in cloud-safe runs because NormalizedGED module was merged into core.
"""

import pytest
pytest.skip(
    "Legacy NormalizedGED module was integrated; use core.metrics/GeDIGCore instead.",
    allow_module_level=True,
)
import numpy as np
import networkx as nx
import torch
import pytest
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

from insightspike.algorithms.normalized_ged import (
    NormalizedGED,
    NormalizedGEDResult,
    calculate_normalized_ged,
    calculate_structural_improvement
)


class TestNormalizedGED:
    """Test suite for NormalizedGED."""
    
    def test_initialization(self):
        """Test proper initialization."""
        calculator = NormalizedGED(
            node_cost=2.0,
            edge_cost=1.5,
            efficiency_weight=0.4,
            normalize_by='max'
        )
        
        assert calculator.node_cost == 2.0
        assert calculator.edge_cost == 1.5
        assert calculator.efficiency_weight == 0.4
        assert calculator.normalize_by == 'max'
    
    def test_simple_ged_calculation(self):
        """Test basic GED calculation."""
        # Graph 1: Triangle
        g1 = nx.Graph()
        g1.add_edges_from([(0, 1), (1, 2), (2, 0)])
        
        # Graph 2: Square
        g2 = nx.Graph()
        g2.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
        
        calculator = NormalizedGED()
        result = calculator.calculate(g1, g2)
        
        assert isinstance(result, NormalizedGEDResult)
        assert result.raw_ged > 0  # Need to add node and edges
        assert 0 <= result.normalized_ged <= 1.0
        assert result.computation_time > 0
    
    def test_identical_graphs(self):
        """Test GED for identical graphs."""
        g1 = nx.karate_club_graph()
        g2 = g1.copy()
        
        calculator = NormalizedGED()
        result = calculator.calculate(g1, g2)
        
        assert result.raw_ged == 0.0
        assert result.normalized_ged == 0.0
        assert result.efficiency_change == 0.0
    
    def test_normalization_methods(self):
        """Test different normalization methods."""
        # Use very different graphs to ensure different normalization factors
        g1 = nx.path_graph(10)  # 10 nodes, 9 edges
        g2 = nx.cycle_graph(5)   # 5 nodes, 5 edges
        
        # Test sum normalization
        calc_sum = NormalizedGED(normalize_by='sum')
        result_sum = calc_sum.calculate(g1, g2)
        
        # Test max normalization
        calc_max = NormalizedGED(normalize_by='max')
        result_max = calc_max.calculate(g1, g2)
        
        # Test average normalization
        calc_avg = NormalizedGED(normalize_by='average')
        result_avg = calc_avg.calculate(g1, g2)
        
        # All should produce valid normalized values
        assert 0 <= result_sum.normalized_ged <= 1.0
        assert 0 <= result_max.normalized_ged <= 1.0
        assert 0 <= result_avg.normalized_ged <= 1.0
        
        # Different methods should produce different factors
        # Sum should be largest, average should be smallest
        assert result_avg.normalization_factor < result_max.normalization_factor < result_sum.normalization_factor
    
    def test_structural_improvement(self):
        """Test structural improvement calculation."""
        # Inefficient structure: path
        g1 = nx.path_graph(10)
        
        # Efficient structure: star
        g2 = nx.star_graph(9)
        
        calculator = NormalizedGED()
        result = calculator.calculate(g1, g2)
        
        # Should show efficiency improvement
        assert result.efficiency_change > 0
        assert result.structural_improvement < 0  # Negative indicates improvement
    
    def test_pyg_data_input(self):
        """Test with PyTorch Geometric Data input."""
        # Create PyG data objects
        data1 = Data(
            x=torch.randn(4, 10),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        )
        
        data2 = Data(
            x=torch.randn(5, 10),
            edge_index=torch.tensor([
                [0, 1, 2, 3, 4, 4, 4, 4],
                [1, 2, 3, 0, 0, 1, 2, 3]
            ], dtype=torch.long)
        )
        
        calculator = NormalizedGED()
        result = calculator.calculate(data1, data2)
        
        assert isinstance(result, NormalizedGEDResult)
        assert result.raw_ged > 0  # Added node and edges
        assert result.ged_value == result.normalized_ged  # Test property
    
    def test_graph_efficiency_calculation(self):
        """Test graph efficiency metric."""
        calculator = NormalizedGED()
        
        # Complete graph - high efficiency
        complete = nx.complete_graph(5)
        eff_complete = calculator._graph_efficiency(complete)
        
        # Path graph - low efficiency
        path = nx.path_graph(5)
        eff_path = calculator._graph_efficiency(path)
        
        # Star graph - medium efficiency
        star = nx.star_graph(4)
        eff_star = calculator._graph_efficiency(star)
        
        # Efficiency ordering
        assert eff_path < eff_star < eff_complete
        assert 0 <= eff_path <= 1
        assert 0 <= eff_complete <= 1
    
    def test_scale_invariance(self):
        """Test that normalization provides scale invariance."""
        # Small transformation
        g1_small = nx.cycle_graph(4)
        g2_small = nx.star_graph(3)
        
        # Same transformation pattern but larger
        g1_large = nx.cycle_graph(8)
        g2_large = nx.star_graph(7)
        
        calculator = NormalizedGED(normalize_by='sum')
        
        result_small = calculator.calculate(g1_small, g2_small)
        result_large = calculator.calculate(g1_large, g2_large)
        
        # Raw GED should be different
        assert result_small.raw_ged != result_large.raw_ged
        
        # But normalized GED should be similar (same transformation pattern)
        assert abs(result_small.normalized_ged - result_large.normalized_ged) < 0.3
    
    def test_edge_cases(self):
        """Test edge cases."""
        calculator = NormalizedGED()
        
        # Empty graphs
        g1 = nx.Graph()
        g2 = nx.Graph()
        result = calculator.calculate(g1, g2)
        assert result.raw_ged == 0.0
        assert result.normalized_ged == 0.0
        
        # Empty to non-empty
        g3 = nx.complete_graph(3)
        result2 = calculator.calculate(g1, g3)
        assert result2.raw_ged > 0
        assert result2.normalized_ged > 0
        
        # Single node graphs
        g4 = nx.Graph()
        g4.add_node(0)
        g5 = nx.Graph()
        g5.add_node(0)
        result3 = calculator.calculate(g4, g5)
        assert result3.raw_ged == 0.0
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        g1 = nx.path_graph(5)
        g2 = nx.star_graph(4)
        
        # Test normalized GED function
        ged = calculate_normalized_ged(g1, g2, normalize_by='max')
        assert isinstance(ged, float)
        assert 0 <= ged <= 1.0
        
        # Test structural improvement function
        improvement = calculate_structural_improvement(g1, g2, efficiency_weight=0.5)
        assert isinstance(improvement, float)
        assert -1.0 <= improvement <= 1.0
    
    def test_numpy_array_input(self):
        """Test with adjacency matrix input."""
        # Create adjacency matrices
        adj1 = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ])
        
        adj2 = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ])
        
        calculator = NormalizedGED()
        result = calculator.calculate(adj1, adj2)
        
        assert isinstance(result, NormalizedGEDResult)
        assert result.raw_ged > 0  # Different edge structure
    
    def test_hub_formation_detection(self):
        """Test detection of hub formation (important for spike detection)."""
        # Distributed structure
        g1 = nx.cycle_graph(6)
        
        # Hub structure
        g2 = g1.copy()
        g2.add_node(6)  # Central hub
        for i in range(6):
            g2.add_edge(6, i)
        
        calculator = NormalizedGED(efficiency_weight=0.3)
        result = calculator.calculate(g1, g2)
        
        # Should detect improvement
        assert result.efficiency_change > 0.1
        assert result.structural_improvement < 0  # Negative = improvement
    
    def test_performance_on_large_graphs(self):
        """Test performance on larger graphs."""
        import time
        
        # Create larger graphs
        g1 = nx.barabasi_albert_graph(100, 3)
        g2 = nx.watts_strogatz_graph(100, 6, 0.3)
        
        calculator = NormalizedGED()
        
        start = time.time()
        result = calculator.calculate(g1, g2)
        elapsed = time.time() - start
        
        # Should complete quickly
        assert elapsed < 0.5  # Less than 0.5 seconds
        assert isinstance(result, NormalizedGEDResult)
        assert 0 <= result.normalized_ged <= 1.0
    
    def test_common_edge_calculation(self):
        """Test accurate common edge counting."""
        # Graph with specific edges
        g1 = nx.Graph()
        g1.add_edges_from([(0, 1), (1, 2), (2, 3)])
        
        # Overlapping graph
        g2 = nx.Graph()
        g2.add_edges_from([(0, 1), (1, 2), (3, 4)])  # 2 common, 1 different
        
        calculator = NormalizedGED()
        result = calculator.calculate(g1, g2)
        
        # Should account for edge differences correctly
        assert result.raw_ged > 0
        assert result.raw_ged == (1 * calculator.node_cost +  # 1 node difference
                                  2 * calculator.edge_cost)    # 2 edge operations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
