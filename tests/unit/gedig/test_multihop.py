"""Unit tests for gedig.multihop module."""

import networkx as nx
import numpy as np
import pytest
import time

from insightspike.algorithms.gedig.multihop import (
    calculate_multihop,
    _compute_hop_sp_gain,
    _default_ged,
    _default_ig,
)
from insightspike.algorithms.gedig.types import GeDIGResult, HopResult


class TestCalculateMultihop:
    """Tests for calculate_multihop function."""

    def test_basic_calculation(self):
        """Test basic multi-hop calculation."""
        g1 = nx.Graph()
        g1.add_edges_from([('a', 'b'), ('b', 'c')])
        g2 = g1.copy()
        g2.add_edge('a', 'c')  # Add shortcut

        features_before = np.random.rand(3, 4).astype(np.float32)
        features_after = np.random.rand(3, 4).astype(np.float32)

        result = calculate_multihop(
            g1, g2,
            features_before, features_after,
            focal_nodes={'a'},
            start_time=time.time(),
            max_hops=2,
        )

        assert isinstance(result, GeDIGResult)
        assert result.gedig_value is not None
        assert len(result.hop_results) > 0

    def test_empty_focal_nodes(self):
        """Test with empty focal nodes."""
        g1 = nx.Graph()
        g1.add_edge('a', 'b')
        g2 = g1.copy()

        result = calculate_multihop(
            g1, g2,
            np.array([]), np.array([]),
            focal_nodes=set(),
            start_time=time.time(),
            max_hops=2,
        )

        # Should return empty result
        assert result.gedig_value == 0.0
        assert len(result.hop_results) == 0

    def test_with_linkset_metrics(self):
        """Test with precomputed linkset metrics."""
        from insightspike.algorithms.gedig.types import LinksetMetrics

        g1 = nx.Graph()
        g1.add_edge('a', 'b')
        g2 = g1.copy()

        linkset_metrics = LinksetMetrics(
            delta_ged_norm=0.1,
            delta_h_norm=0.2,
            delta_sp_rel=0.0,
            gedig_value=-0.1,
            raw_ged=1.0,
            ged_norm_den=10.0,
            ig_norm_den=1.0,
            entropy_before=0.5,
            entropy_after=0.7,
            ig_delta=0.2,
            before_size=2,
            after_size=3,
            query_similarity=1.0,
            pos_w_before=2,
            pos_w_after=3,
            topw_before=[0.8, 0.6],
            topw_after=[0.9, 0.7, 0.5],
        )

        result = calculate_multihop(
            g1, g2,
            np.array([]), np.array([]),
            focal_nodes={'a'},
            start_time=time.time(),
            ig_source_mode='linkset',
            linkset_metrics=linkset_metrics,
        )

        assert result.gedig_value is not None

    def test_adaptive_hops_termination(self):
        """Test adaptive hop termination."""
        # Create identical graphs so gedig is very small
        g1 = nx.Graph()
        g1.add_edge('a', 'b')
        g2 = g1.copy()

        result = calculate_multihop(
            g1, g2,
            np.zeros((2, 4)), np.zeros((2, 4)),
            focal_nodes={'a'},
            start_time=time.time(),
            max_hops=5,
            adaptive_hops=True,
        )

        # With identical graphs, should terminate early
        assert len(result.hop_results) <= 2

    def test_with_custom_ged_calculator(self):
        """Test with custom GED calculator callback."""
        g1 = nx.Graph()
        g1.add_edge('a', 'b')
        g2 = g1.copy()

        custom_ged_called = [False]

        def custom_ged(g1, g2, **kwargs):
            custom_ged_called[0] = True
            return {
                'raw_ged': 0.5,
                'normalized_ged': 0.1,
                'normalization_den': 5.0,
                'structural_cost': 0.1,
                'structural_improvement': -0.1,
            }

        result = calculate_multihop(
            g1, g2,
            np.array([]), np.array([]),
            focal_nodes={'a'},
            start_time=time.time(),
            ged_calculator=custom_ged,
        )

        assert custom_ged_called[0] is True

    def test_with_sp_gain(self):
        """Test multi-hop with SP gain enabled."""
        g1 = nx.Graph()
        g1.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd')])
        g2 = g1.copy()
        g2.add_edge('a', 'd')  # Add shortcut

        result = calculate_multihop(
            g1, g2,
            np.random.rand(4, 4), np.random.rand(4, 4),
            focal_nodes={'a'},
            start_time=time.time(),
            max_hops=2,
            use_multihop_sp_gain=True,
            sp_beta=0.2,
        )

        # Check that hop > 0 has SP computed
        if 1 in result.hop_results:
            assert result.hop_results[1].sp is not None


class TestComputeHopSpGain:
    """Tests for _compute_hop_sp_gain helper."""

    def test_basic_sp_gain(self):
        g1 = nx.Graph()
        g1.add_edges_from([('a', 'b'), ('b', 'c')])
        g2 = g1.copy()
        g2.add_edge('a', 'c')

        delta_sp, sp_mult = _compute_hop_sp_gain(
            g1, g2,
            focal_nodes={'a'},
            hop=1,
            sp_beta=0.2,
        )

        assert isinstance(delta_sp, float)
        assert sp_mult == 0.2

    def test_union_scope_mode(self):
        g1 = nx.Graph()
        g1.add_edges_from([('a', 'b'), ('b', 'c')])
        g2 = g1.copy()
        g2.add_node('d')

        delta_sp, sp_mult = _compute_hop_sp_gain(
            g1, g2,
            focal_nodes={'a'},
            hop=1,
            sp_scope_mode='union',
        )

        assert isinstance(delta_sp, float)

    def test_trim_boundary_mode(self):
        g1 = nx.Graph()
        g1.add_edges_from([('a', 'b'), ('b', 'c')])
        g2 = g1.copy()

        delta_sp, sp_mult = _compute_hop_sp_gain(
            g1, g2,
            focal_nodes={'a'},
            hop=1,
            sp_boundary_mode='trim',
        )

        assert isinstance(delta_sp, float)


class TestDefaultGed:
    """Tests for _default_ged helper."""

    def test_identical_graphs(self):
        g = nx.Graph()
        g.add_edge('a', 'b')

        result = _default_ged(g, g, 1.0, 1.0, 2.0)

        assert result['raw_ged'] == 0.0
        assert result['normalized_ged'] == 0.0

    def test_added_node(self):
        g1 = nx.Graph()
        g1.add_edge('a', 'b')
        g2 = g1.copy()
        g2.add_node('c')

        result = _default_ged(g1, g2, 1.0, 1.0, 2.0)

        assert result['raw_ged'] == 1.0  # One node added
        assert result['normalized_ged'] == 0.5

    def test_added_edge(self):
        g1 = nx.Graph()
        g1.add_nodes_from(['a', 'b', 'c'])
        g1.add_edge('a', 'b')
        g2 = g1.copy()
        g2.add_edge('b', 'c')

        result = _default_ged(g1, g2, 1.0, 1.0, 2.0)

        assert result['raw_ged'] == 1.0  # One edge added


class TestDefaultIg:
    """Tests for _default_ig helper."""

    def test_empty_features(self):
        g = nx.Graph()

        result = _default_ig(g, np.array([]), np.array([]), 1e-10, 2, None)

        assert result['ig_value'] == 0.0
        assert result['entropy_before'] == 0.0
        assert result['entropy_after'] == 0.0

    def test_with_features(self):
        g = nx.Graph()
        g.add_nodes_from([0, 1, 2])

        features_before = np.random.rand(3, 4).astype(np.float32)
        features_after = np.random.rand(3, 4).astype(np.float32)

        result = _default_ig(g, features_before, features_after, 1e-10, 2, None)

        assert 'ig_value' in result
        assert 'entropy_before' in result
        assert 'entropy_after' in result
        assert 'normalization_den' in result

    def test_with_fixed_den(self):
        g = nx.Graph()

        result = _default_ig(g, np.array([[1, 0]]), np.array([[0, 1]]), 1e-10, 2, 2.0)

        assert result['normalization_den'] == 2.0


class TestMultihopEdgeCases:
    """Edge case tests for multihop."""

    def test_single_node_graph(self):
        g1 = nx.Graph()
        g1.add_node('a')
        g2 = g1.copy()

        result = calculate_multihop(
            g1, g2,
            np.array([]), np.array([]),
            focal_nodes={'a'},
            start_time=time.time(),
        )

        assert result.gedig_value is not None

    def test_disconnected_graph(self):
        g1 = nx.Graph()
        g1.add_edge('a', 'b')
        g1.add_edge('c', 'd')  # Disconnected component
        g2 = g1.copy()

        result = calculate_multihop(
            g1, g2,
            np.array([]), np.array([]),
            focal_nodes={'a'},
            start_time=time.time(),
        )

        assert result.gedig_value is not None

    def test_ig_hop0_only_mode(self):
        """Test with ig_hop_apply='hop0'."""
        g1 = nx.Graph()
        g1.add_edges_from([('a', 'b'), ('b', 'c')])
        g2 = g1.copy()

        result = calculate_multihop(
            g1, g2,
            np.random.rand(3, 4), np.random.rand(3, 4),
            focal_nodes={'a'},
            start_time=time.time(),
            max_hops=2,
            ig_source_mode='linkset',
            ig_hop_apply='hop0',
        )

        assert result.gedig_value is not None

    def test_normalized_ig_mode(self):
        """Test with ig_mode='normalized'."""
        g1 = nx.Graph()
        g1.add_edge('a', 'b')
        g2 = g1.copy()

        result = calculate_multihop(
            g1, g2,
            np.random.rand(2, 4), np.random.rand(2, 4),
            focal_nodes={'a'},
            start_time=time.time(),
            ig_mode='normalized',
        )

        assert result.gedig_value is not None

    def test_ig_nonneg_mode(self):
        """Test with ig_nonneg=True."""
        g1 = nx.Graph()
        g1.add_edge('a', 'b')
        g2 = g1.copy()

        result = calculate_multihop(
            g1, g2,
            np.random.rand(2, 4), np.random.rand(2, 4),
            focal_nodes={'a'},
            start_time=time.time(),
            ig_nonneg=True,
        )

        assert result.gedig_value is not None
