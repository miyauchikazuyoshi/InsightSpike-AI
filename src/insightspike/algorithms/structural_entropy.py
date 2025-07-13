"""
Structural Entropy Measures for Graph Analysis
==============================================

Implements various entropy measures for graph structure analysis,
as recommended in the Layer 3 improvements review.

These measures help quantify the information content and organization
of graph structures, complementing the content-based entropy measures.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from collections import Counter
import numpy as np

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

try:
    from torch_geometric.data import Data
    import torch

    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    Data = None
    torch = None

logger = logging.getLogger(__name__)

__all__ = [
    "degree_distribution_entropy",
    "von_neumann_entropy",
    "structural_entropy",
    "clustering_coefficient_entropy",
    "path_length_entropy",
]


def degree_distribution_entropy(graph: Any) -> float:
    """
    Calculate entropy of the degree distribution.

    Higher entropy indicates more uniform degree distribution,
    lower entropy indicates hub-like structures.

    Args:
        graph: NetworkX graph or PyTorch Geometric Data object

    Returns:
        float: Degree distribution entropy in bits
    """
    try:
        # Convert to NetworkX if needed
        nx_graph = _to_networkx(graph)
        if nx_graph is None or nx_graph.number_of_nodes() == 0:
            return 0.0

        # Calculate degree distribution
        degrees = [nx_graph.degree(n) for n in nx_graph.nodes()]

        # Count occurrences of each degree
        degree_counts = Counter(degrees)
        total = sum(degree_counts.values())

        # Calculate probabilities
        probs = [count / total for count in degree_counts.values()]

        # Shannon entropy
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)

        return float(entropy)

    except Exception as e:
        logger.error(f"Degree distribution entropy calculation failed: {e}")
        return 0.0


def von_neumann_entropy(graph: Any, normalized: bool = True) -> float:
    """
    Calculate Von Neumann entropy of the graph.

    This is a spectral entropy measure based on the eigenvalues
    of the graph Laplacian. It captures global structural properties.

    Args:
        graph: NetworkX graph or PyTorch Geometric Data object
        normalized: Whether to use normalized Laplacian

    Returns:
        float: Von Neumann entropy
    """
    try:
        # Convert to NetworkX if needed
        nx_graph = _to_networkx(graph)
        if nx_graph is None or nx_graph.number_of_nodes() < 2:
            return 0.0

        # Calculate Laplacian matrix
        if normalized:
            laplacian = nx.normalized_laplacian_matrix(nx_graph).todense()
        else:
            laplacian = nx.laplacian_matrix(nx_graph).todense()

        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvalsh(laplacian)

        # Normalize eigenvalues (density matrix trace = 1)
        eigenvalues = eigenvalues / np.sum(eigenvalues)

        # Von Neumann entropy: -Σ λ log(λ)
        # Filter out zero and negative eigenvalues
        valid_eigenvalues = eigenvalues[eigenvalues > 1e-10]

        if len(valid_eigenvalues) == 0:
            return 0.0

        entropy = -np.sum(valid_eigenvalues * np.log(valid_eigenvalues))

        return float(entropy)

    except Exception as e:
        logger.error(f"Von Neumann entropy calculation failed: {e}")
        return 0.0


def clustering_coefficient_entropy(graph: Any) -> float:
    """
    Calculate entropy based on clustering coefficient distribution.

    Measures how varied the local clustering is across the graph.

    Args:
        graph: NetworkX graph or PyTorch Geometric Data object

    Returns:
        float: Clustering coefficient entropy
    """
    try:
        # Convert to NetworkX if needed
        nx_graph = _to_networkx(graph)
        if nx_graph is None or nx_graph.number_of_nodes() < 3:
            return 0.0

        # Calculate clustering coefficients
        clustering = nx.clustering(nx_graph)

        if not clustering:
            return 0.0

        # Discretize coefficients into bins
        coeffs = list(clustering.values())
        hist, _ = np.histogram(coeffs, bins=10, range=(0, 1))

        # Calculate probabilities
        total = np.sum(hist)
        if total == 0:
            return 0.0

        probs = hist / total

        # Shannon entropy
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)

        return float(entropy)

    except Exception as e:
        logger.error(f"Clustering coefficient entropy calculation failed: {e}")
        return 0.0


def path_length_entropy(graph: Any, sample_size: Optional[int] = 100) -> float:
    """
    Calculate entropy based on shortest path length distribution.

    For large graphs, samples random pairs to estimate the distribution.

    Args:
        graph: NetworkX graph or PyTorch Geometric Data object
        sample_size: Number of node pairs to sample (None for all pairs)

    Returns:
        float: Path length entropy
    """
    try:
        # Convert to NetworkX if needed
        nx_graph = _to_networkx(graph)
        if nx_graph is None or nx_graph.number_of_nodes() < 2:
            return 0.0

        # Get connected components
        if not nx.is_connected(nx_graph):
            # Use largest connected component
            largest_cc = max(nx.connected_components(nx_graph), key=len)
            nx_graph = nx_graph.subgraph(largest_cc)

        nodes = list(nx_graph.nodes())
        n_nodes = len(nodes)

        if n_nodes < 2:
            return 0.0

        path_lengths = []

        # Sample pairs if graph is large
        if sample_size and n_nodes > sample_size:
            for _ in range(sample_size):
                i, j = np.random.choice(n_nodes, 2, replace=False)
                try:
                    length = nx.shortest_path_length(nx_graph, nodes[i], nodes[j])
                    path_lengths.append(length)
                except nx.NetworkXNoPath:
                    pass
        else:
            # Calculate all pairs shortest paths
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    try:
                        length = nx.shortest_path_length(nx_graph, nodes[i], nodes[j])
                        path_lengths.append(length)
                    except nx.NetworkXNoPath:
                        pass

        if not path_lengths:
            return 0.0

        # Calculate distribution
        length_counts = Counter(path_lengths)
        total = sum(length_counts.values())
        probs = [count / total for count in length_counts.values()]

        # Shannon entropy
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)

        return float(entropy)

    except Exception as e:
        logger.error(f"Path length entropy calculation failed: {e}")
        return 0.0


def structural_entropy(
    graph: Any, weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Calculate multiple structural entropy measures.

    Args:
        graph: NetworkX graph or PyTorch Geometric Data object
        weights: Optional weights for combining measures

    Returns:
        Dictionary with entropy measures
    """
    if weights is None:
        weights = {"degree": 0.4, "clustering": 0.3, "path_length": 0.3}

    measures = {
        "degree_entropy": degree_distribution_entropy(graph),
        "clustering_entropy": clustering_coefficient_entropy(graph),
        "path_length_entropy": path_length_entropy(graph),
    }

    # Calculate weighted combination
    total_weight = sum(weights.values())
    if total_weight > 0:
        measures["combined"] = (
            sum(
                weights.get(key.replace("_entropy", ""), 0) * value
                for key, value in measures.items()
                if key != "combined"
            )
            / total_weight
        )
    else:
        measures["combined"] = np.mean(list(measures.values()))

    return measures


def _to_networkx(graph: Any) -> Optional[nx.Graph]:
    """Convert various graph formats to NetworkX."""
    if graph is None:
        return None

    # Already NetworkX
    if NETWORKX_AVAILABLE and isinstance(graph, nx.Graph):
        return graph

    # PyTorch Geometric Data
    if TORCH_GEOMETRIC_AVAILABLE and isinstance(graph, Data):
        try:
            # Create NetworkX graph from edge_index
            G = nx.Graph()

            # Add nodes
            if hasattr(graph, "num_nodes"):
                G.add_nodes_from(range(graph.num_nodes))
            elif hasattr(graph, "x"):
                G.add_nodes_from(range(graph.x.size(0)))

            # Add edges
            if hasattr(graph, "edge_index"):
                edge_list = graph.edge_index.cpu().numpy().T
                G.add_edges_from(edge_list)

            return G
        except Exception as e:
            logger.error(f"Failed to convert PyTorch Geometric graph: {e}")
            return None

    # Try to create from edge list
    if hasattr(graph, "__iter__"):
        try:
            G = nx.Graph()
            G.add_edges_from(graph)
            return G
        except:
            pass

    logger.warning(f"Unsupported graph type: {type(graph)}")
    return None
