"""Common test helper functions and utilities."""

import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


def assert_graphs_equal(
    g1: nx.Graph, g2: nx.Graph, check_attributes: bool = False
) -> None:
    """Assert that two graphs are equal."""
    assert set(g1.nodes()) == set(g2.nodes()), "Nodes don't match"
    assert set(g1.edges()) == set(g2.edges()), "Edges don't match"

    if check_attributes:
        for node in g1.nodes():
            assert (
                g1.nodes[node] == g2.nodes[node]
            ), f"Node {node} attributes don't match"
        for edge in g1.edges():
            assert (
                g1.edges[edge] == g2.edges[edge]
            ), f"Edge {edge} attributes don't match"


def assert_graphs_similar(
    g1: nx.Graph, g2: nx.Graph, node_tolerance: int = 2, edge_tolerance: int = 3
) -> None:
    """Assert that two graphs are similar within tolerance."""
    node_diff = abs(len(g1.nodes()) - len(g2.nodes()))
    edge_diff = abs(len(g1.edges()) - len(g2.edges()))

    assert (
        node_diff <= node_tolerance
    ), f"Node count differs by {node_diff}, max allowed: {node_tolerance}"
    assert (
        edge_diff <= edge_tolerance
    ), f"Edge count differs by {edge_diff}, max allowed: {edge_tolerance}"


def assert_embeddings_similar(
    emb1: np.ndarray, emb2: np.ndarray, cosine_threshold: float = 0.9
) -> None:
    """Assert that two embeddings are similar based on cosine similarity."""
    # Normalize embeddings
    emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-10)
    emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-10)

    # Calculate cosine similarity
    cosine_sim = np.dot(emb1_norm, emb2_norm)

    assert (
        cosine_sim >= cosine_threshold
    ), f"Cosine similarity {cosine_sim} below threshold {cosine_threshold}"


def create_test_embedding(
    dim: int = 384, normalize: bool = True, seed: Optional[int] = None
) -> np.ndarray:
    """Create a test embedding with optional normalization."""
    if seed is not None:
        np.random.seed(seed)

    embedding = np.random.randn(dim).astype(np.float32)

    if normalize:
        embedding = embedding / np.linalg.norm(embedding)

    return embedding


def create_embedding_batch(
    num_samples: int, dim: int = 384, normalize: bool = True
) -> np.ndarray:
    """Create a batch of test embeddings."""
    embeddings = np.random.randn(num_samples, dim).astype(np.float32)

    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-10)

    return embeddings


@contextmanager
def assert_execution_time(max_seconds: float):
    """Context manager to assert execution time is within limit."""
    start_time = time.time()
    yield
    elapsed = time.time() - start_time
    assert (
        elapsed <= max_seconds
    ), f"Execution took {elapsed:.2f}s, max allowed: {max_seconds}s"


def calculate_graph_similarity(g1: nx.Graph, g2: nx.Graph) -> float:
    """Calculate similarity between two graphs (0-1 scale)."""
    # Simple Jaccard similarity on nodes and edges
    nodes1, nodes2 = set(g1.nodes()), set(g2.nodes())
    edges1, edges2 = set(g1.edges()), set(g2.edges())

    node_intersection = len(nodes1 & nodes2)
    node_union = len(nodes1 | nodes2)
    node_similarity = node_intersection / node_union if node_union > 0 else 0

    edge_intersection = len(edges1 & edges2)
    edge_union = len(edges1 | edges2)
    edge_similarity = edge_intersection / edge_union if edge_union > 0 else 0

    # Weighted average
    return 0.4 * node_similarity + 0.6 * edge_similarity


def generate_test_insights(num_insights: int = 5) -> List[Dict[str, Any]]:
    """Generate test insight data."""
    insights = []
    categories = ["pattern", "relationship", "anomaly", "trend", "correlation"]

    for i in range(num_insights):
        insight = {
            "id": f"insight_{i}",
            "category": categories[i % len(categories)],
            "confidence": np.random.rand(),
            "timestamp": time.time() + i,
            "description": f"Test insight {i}: Discovered {categories[i % len(categories)]}",
            "delta_ged": np.random.uniform(-1, 1),
            "delta_ig": np.random.uniform(-0.5, 0.5),
            "is_spike": np.random.rand() > 0.7,
        }
        insights.append(insight)

    return insights


def validate_episode_format(episode: Any) -> bool:
    """Validate that an episode has the required format."""
    required_attrs = ["text", "timestamp", "embedding"]

    for attr in required_attrs:
        if not hasattr(episode, attr):
            return False

    # Check embedding is numpy array
    if not isinstance(episode.embedding, np.ndarray):
        return False

    # Check timestamp is numeric
    try:
        float(episode.timestamp)
    except (TypeError, ValueError):
        return False

    return True


def create_mock_config_object():
    """Create a mock configuration object matching the real Config structure."""
    from unittest.mock import Mock

    config = Mock()

    # Embedding config
    config.embedding = Mock()
    config.embedding.model_name = "test-model"
    config.embedding.dimension = 384
    config.embedding.device = "cpu"

    # LLM config
    config.llm = Mock()
    config.llm.provider = "mock"
    config.llm.model_name = "mock-llm"
    config.llm.safe_mode = True

    # Memory config
    config.memory = Mock()
    config.memory.max_retrieved_docs = 10
    config.memory.merge_ged = 0.4
    config.memory.split_ig = -0.15

    # Reasoning config
    config.reasoning = Mock()
    config.reasoning.use_gnn = False
    config.reasoning.spike_ged_threshold = 0.5
    config.reasoning.spike_ig_threshold = 0.2
    config.reasoning.use_scalable_graph = True

    # GNN property for backward compatibility
    config.gnn = config.reasoning

    return config


def assert_metric_in_range(
    value: float, expected: float, tolerance: float = 0.1, metric_name: str = "metric"
) -> None:
    """Assert that a metric value is within expected range."""
    lower = expected - tolerance
    upper = expected + tolerance
    assert lower <= value <= upper, (
        f"{metric_name} value {value} outside expected range "
        f"[{lower}, {upper}] (expected: {expected} ± {tolerance})"
    )


def create_test_layer_input(
    num_documents: int = 5, include_graph: bool = True
) -> Dict[str, Any]:
    """Create test input for layer processing."""
    from tests.factories.mock_factory import document_factory, graph_factory

    layer_input = {
        "documents": document_factory.create_test_documents(num_documents),
        "query": "Test query for processing",
        "metadata": {"timestamp": time.time(), "session_id": "test_session_123"},
    }

    if include_graph:
        layer_input["context_graph"] = graph_factory.create_simple_graph(num_documents)

    return layer_input


# Performance testing helpers
class PerformanceMetrics:
    """Helper for tracking performance metrics during tests."""

    def __init__(self):
        self.metrics = {}

    def record(self, name: str, value: float) -> None:
        """Record a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get_average(self, name: str) -> float:
        """Get average value for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        return np.mean(self.metrics[name])

    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        values = np.array(self.metrics[name])
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
        }

    def assert_performance(
        self, name: str, max_mean: float, max_std: Optional[float] = None
    ) -> None:
        """Assert performance metrics are within bounds."""
        stats = self.get_stats(name)
        assert (
            stats["mean"] <= max_mean
        ), f"{name} mean {stats['mean']:.3f} exceeds max {max_mean}"

        if max_std is not None:
            assert (
                stats["std"] <= max_std
            ), f"{name} std {stats['std']:.3f} exceeds max {max_std}"
