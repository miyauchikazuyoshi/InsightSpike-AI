"""Unit tests for gedig.graph_utils module."""

import pytest
import networkx as nx
import numpy as np

from insightspike.algorithms.gedig.graph_utils import (
    graph_efficiency,
    spectral_score,
    avg_shortest_path_length_safe,
    compute_sp_gain_norm,
    extract_k_hop_subgraph,
    trim_terminal_edges,
    ensure_networkx,
    pyg_to_networkx,
    extract_features,
    filter_features,
    compute_ged_min_proxy,
)


class TestGraphEfficiency:
    """Tests for graph_efficiency function."""

    def test_empty_graph(self):
        g = nx.Graph()
        assert graph_efficiency(g) == 0.0

    def test_single_node(self):
        g = nx.Graph()
        g.add_node(1)
        assert graph_efficiency(g) == 0.0

    def test_path_graph(self):
        g = nx.path_graph(5)
        eff = graph_efficiency(g)
        assert 0 < eff < 1

    def test_complete_graph(self):
        g = nx.complete_graph(5)
        eff = graph_efficiency(g)
        # Complete graph has high efficiency
        assert eff > 0.5


class TestSpectralScore:
    """Tests for spectral_score function."""

    def test_empty_graph(self):
        g = nx.Graph()
        assert spectral_score(g) == 0.0

    def test_single_node(self):
        g = nx.Graph()
        g.add_node(1)
        assert spectral_score(g) == 0.0

    def test_path_graph(self):
        g = nx.path_graph(5)
        score = spectral_score(g)
        assert score >= 0.0

    def test_complete_graph(self):
        g = nx.complete_graph(5)
        score = spectral_score(g)
        assert score >= 0.0


class TestAvgShortestPathLengthSafe:
    """Tests for avg_shortest_path_length_safe function."""

    def test_empty_graph(self):
        g = nx.Graph()
        assert avg_shortest_path_length_safe(g) == 0.0

    def test_single_node(self):
        g = nx.Graph()
        g.add_node(1)
        assert avg_shortest_path_length_safe(g) == 0.0

    def test_path_graph(self):
        g = nx.path_graph(5)
        length = avg_shortest_path_length_safe(g)
        # Average path length for path graph
        assert 1.0 < length < 3.0

    def test_disconnected_graph(self):
        g = nx.Graph()
        g.add_edge(1, 2)
        g.add_edge(3, 4)  # Disconnected component
        length = avg_shortest_path_length_safe(g)
        # Should handle disconnected graphs
        assert length >= 0.0


class TestComputeSpGainNorm:
    """Tests for compute_sp_gain_norm function."""

    def test_identical_graphs(self):
        g = nx.path_graph(5)
        gain = compute_sp_gain_norm(g, g)
        assert abs(gain) < 0.01

    def test_added_shortcut(self):
        g1 = nx.path_graph(5)
        g2 = g1.copy()
        g2.add_edge(0, 4)  # Add shortcut
        gain = compute_sp_gain_norm(g1, g2)
        # Adding shortcut should decrease path length (positive gain)
        assert gain > 0

    def test_empty_graph(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        gain = compute_sp_gain_norm(g1, g2)
        assert gain == 0.0


class TestExtractKHopSubgraph:
    """Tests for extract_k_hop_subgraph function."""

    def test_zero_hop(self):
        g = nx.path_graph(5)
        focal = {"1", "2"}
        # Convert to string nodes
        g = nx.relabel_nodes(g, {i: str(i) for i in range(5)})
        sub, nodes = extract_k_hop_subgraph(g, focal, k=0)
        assert nodes == focal

    def test_one_hop(self):
        g = nx.path_graph(5)
        g = nx.relabel_nodes(g, {i: str(i) for i in range(5)})
        focal = {"2"}
        sub, nodes = extract_k_hop_subgraph(g, focal, k=1)
        assert "1" in nodes
        assert "2" in nodes
        assert "3" in nodes
        assert "0" not in nodes
        assert "4" not in nodes

    def test_empty_focal(self):
        g = nx.path_graph(5)
        sub, nodes = extract_k_hop_subgraph(g, set(), k=2)
        assert len(nodes) == 0
        assert sub.number_of_nodes() == 0

    def test_invalid_focal_nodes(self):
        g = nx.path_graph(5)
        focal = {"999", "888"}
        sub, nodes = extract_k_hop_subgraph(g, focal, k=2)
        assert len(nodes) == 0


class TestTrimTerminalEdges:
    """Tests for trim_terminal_edges function."""

    def test_basic_trim(self):
        g = nx.path_graph(5)
        g = nx.relabel_nodes(g, {i: str(i) for i in range(5)})
        anchors = {"2"}
        result = trim_terminal_edges(g, anchors, hop=2)
        # Should trim edges at hop distance 2
        assert result.number_of_nodes() == g.number_of_nodes()

    def test_empty_anchors(self):
        g = nx.path_graph(5)
        result = trim_terminal_edges(g, set(), hop=2)
        # Should return copy without changes
        assert result.number_of_edges() == g.number_of_edges()


class TestEnsureNetworkx:
    """Tests for ensure_networkx function."""

    def test_networkx_passthrough(self):
        g = nx.path_graph(5)
        result = ensure_networkx(g)
        assert result is g

    def test_numpy_adjacency(self):
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        result = ensure_networkx(adj)
        assert isinstance(result, nx.Graph)
        assert result.number_of_nodes() == 3

    def test_invalid_input(self):
        result = ensure_networkx("invalid")
        assert isinstance(result, nx.Graph)
        assert result.number_of_nodes() == 0


class TestPygToNetworkx:
    """Tests for pyg_to_networkx function."""

    def test_mock_pyg_data(self):
        # Create a mock PyG-like object
        class MockPyGData:
            def __init__(self):
                self.num_nodes = 3
                self.edge_index = np.array([[0, 1, 2], [1, 2, 0]])
                self.x = np.random.randn(3, 64)

        data = MockPyGData()
        g = pyg_to_networkx(data)
        assert isinstance(g, nx.Graph)
        assert g.number_of_nodes() == 3

    def test_empty_data(self):
        class EmptyData:
            pass

        data = EmptyData()
        g = pyg_to_networkx(data)
        assert g.number_of_nodes() == 0


class TestExtractFeatures:
    """Tests for extract_features function."""

    def test_with_features(self):
        g = nx.Graph()
        g.add_node(0, feature=np.array([1.0, 2.0, 3.0]))
        g.add_node(1, feature=np.array([4.0, 5.0, 6.0]))
        features = extract_features(g)
        assert features.shape == (2, 3)

    def test_with_vec_attribute(self):
        g = nx.Graph()
        g.add_node(0, vec=np.array([1.0, 2.0]))
        g.add_node(1, vec=np.array([3.0, 4.0]))
        features = extract_features(g)
        assert features.shape == (2, 2)

    def test_without_features(self):
        g = nx.path_graph(3)
        features = extract_features(g)
        assert features.shape[0] == 3
        assert features.shape[1] == 64  # Default random features


class TestFilterFeatures:
    """Tests for filter_features function."""

    def test_basic_filter(self):
        g = nx.Graph()
        g.add_node("a")
        g.add_node("b")
        g.add_node("c")
        features = np.array([[1, 2], [3, 4], [5, 6]])
        node_set = {"a", "c"}
        filtered = filter_features(features, node_set, g)
        assert filtered.shape[0] == 2

    def test_empty_node_set(self):
        g = nx.path_graph(3)
        features = np.array([[1, 2], [3, 4], [5, 6]])
        filtered = filter_features(features, set(), g)
        assert filtered.shape[0] == 0


class TestComputeGedMinProxy:
    """Tests for compute_ged_min_proxy function."""

    def test_identical_graphs(self):
        g = nx.path_graph(5)
        proxy = compute_ged_min_proxy(g, g)
        assert proxy == 0.0

    def test_with_shortcut(self):
        g1 = nx.path_graph(5)
        g2 = g1.copy()
        g2.add_edge(0, 4)
        proxy = compute_ged_min_proxy(g1, g2)
        # Path shortening should give positive proxy
        assert proxy > 0

    def test_empty_graphs(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        proxy = compute_ged_min_proxy(g1, g2)
        assert proxy == 0.0
