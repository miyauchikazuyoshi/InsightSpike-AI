"""Graph fixtures for consistent test data across the test suite."""

import pytest
import networkx as nx
import numpy as np
from typing import Tuple, List, Dict, Any


@pytest.fixture
def simple_graph() -> nx.Graph:
    """Create a simple graph for basic testing."""
    g = nx.Graph()
    g.add_nodes_from([1, 2, 3, 4, 5])
    g.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])
    return g


@pytest.fixture
def complex_graph() -> nx.Graph:
    """Create a more complex graph with multiple components."""
    g = nx.Graph()
    # First component
    g.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (2, 4)])
    # Second component
    g.add_edges_from([(5, 6), (6, 7), (7, 8), (8, 5)])
    # Bridge
    g.add_edge(4, 5)
    return g


@pytest.fixture
def empty_graph() -> nx.Graph:
    """Create an empty graph."""
    return nx.Graph()


@pytest.fixture
def single_node_graph() -> nx.Graph:
    """Create a graph with a single node."""
    g = nx.Graph()
    g.add_node(1)
    return g


@pytest.fixture
def disconnected_graph() -> nx.Graph:
    """Create a disconnected graph with multiple components."""
    g = nx.Graph()
    # Component 1
    g.add_edges_from([(1, 2), (2, 3)])
    # Component 2
    g.add_edges_from([(4, 5), (5, 6)])
    # Isolated nodes
    g.add_nodes_from([7, 8, 9])
    return g


@pytest.fixture
def weighted_graph() -> nx.Graph:
    """Create a weighted graph."""
    g = nx.Graph()
    edges_with_weights = [
        (1, 2, 0.5),
        (2, 3, 0.8),
        (3, 4, 0.3),
        (4, 5, 0.9),
        (5, 1, 0.6)
    ]
    g.add_weighted_edges_from(edges_with_weights)
    return g


@pytest.fixture
def graph_pair_similar() -> Tuple[nx.Graph, nx.Graph]:
    """Create a pair of similar graphs (small edit distance)."""
    g1 = nx.Graph()
    g1.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])
    
    g2 = nx.Graph()
    g2.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])  # One extra node
    
    return g1, g2


@pytest.fixture
def graph_pair_different() -> Tuple[nx.Graph, nx.Graph]:
    """Create a pair of very different graphs (large edit distance)."""
    g1 = nx.Graph()
    g1.add_edges_from([(1, 2), (2, 3)])
    
    g2 = nx.Graph()
    g2.add_edges_from([(10, 20), (20, 30), (30, 40), (40, 50), (50, 60)])
    
    return g1, g2


@pytest.fixture
def graph_with_attributes() -> nx.Graph:
    """Create a graph with node and edge attributes."""
    g = nx.Graph()
    
    # Add nodes with attributes
    g.add_node(1, label="A", weight=1.0)
    g.add_node(2, label="B", weight=2.0)
    g.add_node(3, label="C", weight=3.0)
    
    # Add edges with attributes
    g.add_edge(1, 2, weight=0.5, type="strong")
    g.add_edge(2, 3, weight=0.3, type="weak")
    g.add_edge(3, 1, weight=0.8, type="strong")
    
    return g


# Graph data configurations for factory pattern
GRAPH_CONFIGS = {
    "simple": {
        "nodes": [1, 2, 3, 4, 5],
        "edges": [(1, 2), (2, 3), (3, 4), (4, 5)]
    },
    "complex": {
        "nodes": list(range(1, 11)),
        "edges": [(i, i+1) for i in range(1, 10)] + [(10, 1)]  # Cycle
    },
    "tree": {
        "nodes": list(range(1, 8)),
        "edges": [(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7)]  # Binary tree
    },
    "star": {
        "nodes": list(range(1, 7)),
        "edges": [(1, i) for i in range(2, 7)]  # Star topology
    },
    "complete": {
        "nodes": [1, 2, 3, 4],
        "edges": [(i, j) for i in range(1, 5) for j in range(i+1, 5)]  # Complete graph K4
    }
}


def create_graph_from_config(config: Dict[str, Any]) -> nx.Graph:
    """Create a graph from a configuration dictionary."""
    g = nx.Graph()
    g.add_nodes_from(config.get("nodes", []))
    g.add_edges_from(config.get("edges", []))
    return g


@pytest.fixture
def graph_factory():
    """Factory fixture for creating graphs from configurations."""
    def _create(config_name: str) -> nx.Graph:
        if config_name not in GRAPH_CONFIGS:
            raise ValueError(f"Unknown graph configuration: {config_name}")
        return create_graph_from_config(GRAPH_CONFIGS[config_name])
    return _create