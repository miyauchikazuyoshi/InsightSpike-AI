"""Graph utilities for geDIG.

This module provides graph manipulation and analysis functions.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict, Optional, Set, Tuple

import networkx as nx
import numpy as np


def graph_efficiency(g: nx.Graph) -> float:
    """Calculate combined graph efficiency metric.

    Returns a weighted combination of global efficiency and clustering.

    Args:
        g: Input graph.

    Returns:
        Combined efficiency score (0.7 * global_efficiency + 0.3 * clustering).
    """
    if g.number_of_nodes() == 0:
        return 0.0
    try:
        ge = nx.global_efficiency(g)
    except Exception:
        ge = 0.0
    try:
        cl = nx.average_clustering(g)
    except Exception:
        cl = 0.0
    return 0.7 * ge + 0.3 * cl


def spectral_score(g: nx.Graph) -> float:
    """Calculate structural score using Laplacian eigenvalues.

    Returns standard deviation of eigenvalues (higher = more irregular structure).

    Args:
        g: Input graph.

    Returns:
        Standard deviation of Laplacian eigenvalues.
    """
    if g.number_of_nodes() < 2:
        return 0.0
    try:
        L = nx.laplacian_matrix(g).toarray()
        eig = np.linalg.eigvalsh(L)
        return float(np.std(eig))
    except Exception:
        return 0.0


def avg_shortest_path_length_safe(
    g: nx.Graph,
    node_cap: int = 200,
    pair_samples: int = 400,
) -> float:
    """Calculate average shortest-path length over connected pairs.

    Uses exact all-pairs for small graphs; for larger graphs, falls back to
    sampling over node pairs to bound runtime.

    Args:
        g: Input graph.
        node_cap: Node count threshold for exact vs sampled computation.
        pair_samples: Number of pairs to sample for large graphs.

    Returns:
        Average shortest path length, or 0.0 on degenerate cases.
    """
    n = g.number_of_nodes()
    if n < 2:
        return 0.0

    # Exact path lengths for small graphs
    if n <= max(32, node_cap // 3):
        try:
            total = 0
            count = 0
            for u, lengths in nx.all_pairs_shortest_path_length(g):
                for v, d in lengths.items():
                    if v <= u:
                        continue
                    total += d
                    count += 1
            return (total / count) if count > 0 else 0.0
        except Exception:
            return 0.0

    # Sampling for larger graphs
    try:
        nodes = list(g.nodes())
        if len(nodes) < 2:
            return 0.0
        samples = min(pair_samples, (n * (n - 1)) // 2)
        if samples <= 0:
            return 0.0
        total = 0.0
        count = 0
        for _ in range(samples):
            u, v = random.sample(nodes, 2)
            try:
                d = nx.shortest_path_length(g, u, v)
                total += float(d)
                count += 1
            except Exception:
                continue
        return (total / count) if count > 0 else 0.0
    except Exception:
        return 0.0


def compute_sp_gain_norm(
    g_before: nx.Graph,
    g_after: nx.Graph,
    mode: str = 'relative',
    node_cap: int = 200,
    pair_samples: int = 400,
) -> float:
    """Compute normalized signed shortest-path gain between two graphs.

    Args:
        g_before: Graph before change.
        g_after: Graph after change.
        mode: 'relative' for (L_before - L_after) / L_before.
        node_cap: Node count threshold for exact vs sampled computation.
        pair_samples: Number of pairs to sample for large graphs.

    Returns:
        Normalized SP gain in [-1, 1].
    """
    Lb = avg_shortest_path_length_safe(g_before, node_cap, pair_samples)
    La = avg_shortest_path_length_safe(g_after, node_cap, pair_samples)

    if Lb <= 0.0:
        # Before has no connectivity (no edges or disconnected).
        # If after has connectivity, this is an improvement from "no paths" to "paths exist".
        if La > 0.0:
            # Maximum improvement: went from no connectivity to some connectivity.
            return 1.0
        # Both have no connectivity - no change.
        return 0.0

    gain = Lb - La
    if mode == 'relative':
        return max(-1.0, min(1.0, gain / Lb))
    return max(-1.0, min(1.0, gain))


def extract_k_hop_subgraph(
    graph: nx.Graph,
    focal_nodes: Set[str],
    k: int,
) -> Tuple[nx.Graph, Set[str]]:
    """Extract k-hop subgraph around focal nodes.

    Args:
        graph: Source graph.
        focal_nodes: Set of focal node IDs.
        k: Number of hops.

    Returns:
        Tuple of (subgraph, all_nodes_in_subgraph).
    """
    valid = {n for n in focal_nodes if n in graph}
    if not valid:
        return nx.Graph(), set()

    if k == 0:
        return graph.subgraph(valid).copy(), valid

    all_nodes = set(valid)
    current = valid

    for _ in range(k):
        nxt = set()
        for n in current:
            if n in graph:
                nxt.update(graph.neighbors(n))
        all_nodes.update(nxt)
        current = nxt

    return graph.subgraph(all_nodes).copy(), all_nodes


def trim_terminal_edges(
    g: nx.Graph,
    anchors: Set[str],
    hop: int,
) -> nx.Graph:
    """Trim edges incident to terminal layer while keeping nodes.

    Distances are computed from anchors via BFS limited to hop.

    Args:
        g: Input graph.
        anchors: Anchor node set.
        hop: Maximum hop distance.

    Returns:
        Graph with terminal edges removed.
    """
    try:
        dist: Dict[Any, Optional[int]] = {n: None for n in g.nodes()}
        dq: deque = deque()

        for a in anchors:
            if a in g:
                dist[a] = 0
                dq.append(a)

        while dq:
            u = dq.popleft()
            du = dist[u]
            if du is None or du >= hop:
                continue
            for v in g.neighbors(u):
                if dist[v] is None:
                    dist[v] = du + 1
                    if dist[v] < hop:
                        dq.append(v)

        out = g.copy()
        to_remove = []

        for u, v in out.edges():
            du = dist.get(u, None)
            dv = dist.get(v, None)
            if du is None or dv is None:
                continue
            # Only remove edges where BOTH endpoints are at terminal distance.
            # This preserves edges from anchor to terminal nodes while removing
            # boundary edges between terminal nodes.
            if du == hop and dv == hop:
                to_remove.append((u, v))

        if to_remove:
            out.remove_edges_from(to_remove)

        return out
    except Exception:
        return g.copy()


def ensure_networkx(graph: Any) -> nx.Graph:
    """Convert various graph types to NetworkX.

    Args:
        graph: Input graph (NetworkX, PyG Data, or adjacency matrix).

    Returns:
        NetworkX Graph.
    """
    if isinstance(graph, nx.Graph):
        return graph

    # Handle PyG Data
    if hasattr(graph, 'edge_index') or hasattr(graph, 'x'):
        return pyg_to_networkx(graph)

    # Handle adjacency matrix
    if isinstance(graph, np.ndarray) and graph.ndim == 2:
        if graph.shape[0] == graph.shape[1]:
            return nx.from_numpy_array(graph)
        return nx.Graph()

    return nx.Graph()


def pyg_to_networkx(data: Any) -> nx.Graph:
    """Convert PyTorch Geometric Data to NetworkX.

    Args:
        data: PyG Data object.

    Returns:
        NetworkX Graph.
    """
    G = nx.Graph()

    # Add nodes
    if hasattr(data, 'num_nodes'):
        num_nodes = data.num_nodes
    elif hasattr(data, 'x') and data.x is not None:
        num_nodes = data.x.shape[0]
    else:
        return G

    G.add_nodes_from(range(num_nodes))

    # Add edges
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        edge_array = data.edge_index
        if hasattr(edge_array, 'cpu'):
            edge_array = edge_array.cpu().numpy()
        edge_array = np.array(edge_array)
        if edge_array.ndim == 2 and edge_array.shape[0] == 2:
            edges = edge_array.T.tolist()
            G.add_edges_from(edges)

    # Add node features as attributes
    if hasattr(data, 'x') and data.x is not None:
        features = data.x
        if hasattr(features, 'cpu'):
            features = features.cpu().numpy()
        for i in range(num_nodes):
            if i < len(features):
                G.nodes[i]['feature'] = features[i]

    return G


def extract_features(graph: nx.Graph) -> np.ndarray:
    """Extract or generate node features from graph.

    Args:
        graph: Input graph.

    Returns:
        Feature array of shape (num_nodes, feature_dim).
    """
    features = []
    for node in graph.nodes():
        node_data = graph.nodes[node]
        if 'feature' in node_data:
            features.append(node_data['feature'])
        elif 'vec' in node_data:
            features.append(node_data['vec'])
        else:
            features.append(np.random.randn(64))
    return np.array(features)


def filter_features(
    features: np.ndarray,
    node_set: Set[str],
    original_graph: nx.Graph,
) -> np.ndarray:
    """Filter features to match node subset.

    Args:
        features: Full feature array.
        node_set: Set of node IDs to keep.
        original_graph: Original graph for node ordering.

    Returns:
        Filtered feature array.
    """
    node_to_idx = {node: i for i, node in enumerate(original_graph.nodes())}
    filtered = [
        features[node_to_idx[n]]
        for n in sorted(node_set)
        if n in node_to_idx and node_to_idx[n] < len(features)
    ]
    if filtered:
        return np.array(filtered)
    if features.ndim >= 2:
        return np.empty((0, features.shape[1]))
    return np.empty((0,))


def compute_ged_min_proxy(g_before: nx.Graph, g_after: nx.Graph) -> float:
    """Approximate GED_min via relative average shortest-path shortening.

    Uses undirected graphs and the largest connected component to avoid inf.

    Args:
        g_before: Graph before change.
        g_after: Graph after change.

    Returns:
        Relative SP gain or edge densification ratio.
    """
    def _avg_sp(g: nx.Graph) -> float:
        if g.number_of_nodes() < 2:
            return 0.0
        und = g.to_undirected()
        if not nx.is_connected(und):
            comp = max(nx.connected_components(und), key=len)
            und = und.subgraph(comp).copy()
            if und.number_of_nodes() < 2:
                return 0.0
        try:
            return float(nx.average_shortest_path_length(und))
        except Exception:
            return 0.0

    asp_before = _avg_sp(g_before)
    asp_after = _avg_sp(g_after)

    if asp_before > 0.0:
        asp_gain = float((asp_before - asp_after) / max(asp_before, 1.0))
        if asp_gain > 0.0:
            return asp_gain

    # Fallback: edge densification
    e_before = g_before.number_of_edges()
    e_after = g_after.number_of_edges()
    if e_after != e_before:
        return float((e_after - e_before) / max(e_before, 1.0))

    return 0.0


def compute_betti_1(g: nx.Graph) -> int:
    """First Betti number: β₁ = E - V + C.

    Counts independent cycles in the graph.
    For connected graphs (C=1), simplifies to E - V + 1.

    Computational cost: O(V+E) general, O(1) if connected.
    """
    V = g.number_of_nodes()
    if V == 0:
        return 0
    E = g.number_of_edges()
    C = nx.number_connected_components(g)
    return E - V + C


__all__ = [
    "graph_efficiency",
    "spectral_score",
    "avg_shortest_path_length_safe",
    "compute_sp_gain_norm",
    "extract_k_hop_subgraph",
    "trim_terminal_edges",
    "ensure_networkx",
    "pyg_to_networkx",
    "extract_features",
    "filter_features",
    "compute_ged_min_proxy",
    "compute_betti_1",
]
