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

    def test_disconnected_graph(self):
        """Test with disconnected graph (uses largest component)."""
        g1 = nx.Graph()
        g1.add_edges_from([(0, 1), (1, 2), (3, 4)])  # Two components
        g2 = g1.copy()
        g2.add_edge(0, 4)  # Connect components
        proxy = compute_ged_min_proxy(g1, g2)
        assert isinstance(proxy, float)

    def test_edge_densification_fallback(self):
        """Test edge count fallback when SP gain is not positive."""
        g1 = nx.Graph()
        g1.add_node(0)
        g2 = nx.Graph()
        g2.add_edge(0, 1)
        proxy = compute_ged_min_proxy(g1, g2)
        assert proxy != 0.0  # Should use edge fallback


class TestAvgShortestPathLengthSafeLargeGraph:
    """Additional tests for avg_shortest_path_length_safe with large graphs."""

    def test_large_graph_sampling(self):
        """Test sampling behavior on large graphs."""
        # Create a large graph that triggers sampling
        g = nx.watts_strogatz_graph(100, 4, 0.3)
        avg_sp = avg_shortest_path_length_safe(g, node_cap=20, pair_samples=50)
        assert avg_sp > 0

    def test_disconnected_large_graph(self):
        """Test with large disconnected graph."""
        g = nx.Graph()
        for i in range(50):
            g.add_edge(i * 2, i * 2 + 1)  # 50 disconnected pairs
        avg_sp = avg_shortest_path_length_safe(g, node_cap=20, pair_samples=100)
        # Many pairs won't be connected
        assert avg_sp >= 0


class TestComputeSpGainNormAbsolute:
    """Additional tests for compute_sp_gain_norm absolute mode."""

    def test_absolute_mode(self):
        """Test absolute mode."""
        g1 = nx.path_graph(10)
        g2 = g1.copy()
        g2.add_edge(0, 9)  # Add shortcut
        gain = compute_sp_gain_norm(g1, g2, mode='absolute')
        assert gain > 0


class TestTrimTerminalEdgesExtended:
    """Extended tests for trim_terminal_edges."""

    def test_with_terminal_layer(self):
        """Test that boundary edges between two terminal nodes are removed.

        Per the trim_terminal_edges implementation, an edge is removed only
        when BOTH endpoints are at the terminal hop distance. This preserves
        edges from anchor toward terminal nodes while pruning the boundary
        ring between two terminal nodes.
        """
        g = nx.Graph()
        # a -- b -- c1, b -- c2, and c1 -- c2 (boundary at hop=2)
        g.add_edges_from([
            ('a', 'b'), ('b', 'c1'), ('b', 'c2'), ('c1', 'c2')
        ])
        anchors = {'a'}
        trimmed = trim_terminal_edges(g, anchors, hop=2)
        # Both c1 and c2 are at hop=2 (terminal), so the boundary edge between
        # them is removed.
        assert ('c1', 'c2') not in trimmed.edges()
        # Edges from interior to terminal are kept.
        assert ('a', 'b') in trimmed.edges()
        assert ('b', 'c1') in trimmed.edges()
        assert ('b', 'c2') in trimmed.edges()

    def test_exception_handling(self):
        """Test that exceptions are handled gracefully."""
        g = nx.Graph()
        g.add_edge('a', 'b')
        # Empty anchors should still work
        result = trim_terminal_edges(g, set(), hop=2)
        assert isinstance(result, nx.Graph)


class TestEnsureNetworkxExtended:
    """Extended tests for ensure_networkx."""

    def test_non_square_matrix(self):
        """Test with non-square adjacency matrix."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 matrix
        g = ensure_networkx(arr)
        assert g.number_of_nodes() == 0


class TestPygToNetworkxExtended:
    """Extended tests for pyg_to_networkx."""

    def test_with_cpu_method(self):
        """Test PyG data that has cpu() method (tensor-like)."""
        class TensorLike:
            def __init__(self, data):
                self._data = np.array(data)

            def cpu(self):
                return self

            def numpy(self):
                return self._data

            @property
            def shape(self):
                return self._data.shape

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

        class MockPyGData:
            def __init__(self):
                self.num_nodes = 3
                self.edge_index = TensorLike([[0, 1], [1, 2]])
                self.x = TensorLike([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        data = MockPyGData()
        g = pyg_to_networkx(data)
        assert g.number_of_nodes() == 3


class TestFilterFeaturesExtended:
    """Extended tests for filter_features."""

    def test_1d_features_empty_result(self):
        """Test with 1D features returning empty."""
        g = nx.Graph()
        g.add_node("a")
        features = np.array([1.0, 2.0, 3.0])  # 1D
        filtered = filter_features(features, {"z"}, g)  # Non-existent node
        assert filtered.shape == (0,)
