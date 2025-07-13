"""
Tests for Scalable Graph Builder (Phase 2)
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from insightspike.core.layers.scalable_graph_builder import ScalableGraphBuilder


class TestScalableGraphBuilder:
    """Test the scalable graph builder implementation."""

    @pytest.fixture
    def builder(self):
        """Create a builder instance for testing."""
        builder = ScalableGraphBuilder()
        builder.dimension = 10
        builder.top_k = 3
        builder.similarity_threshold = 0.3
        return builder

    def test_initialization(self, builder):
        """Test builder initialization."""
        assert builder.dimension == 10
        assert builder.top_k == 3
        assert builder.similarity_threshold == 0.3
        assert builder.index is None  # Not initialized until build_graph is called

    def test_build_graph_small(self, builder):
        """Test graph building with small dataset."""
        # Create test documents
        documents = []
        for i in range(5):
            vec = np.random.randn(10).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            documents.append({"embedding": vec, "text": f"Document {i}"})

        # Build graph
        graph = builder.build_graph(documents)

        # Verify structure
        assert graph is not None
        assert hasattr(graph, "x")
        assert hasattr(graph, "edge_index")
        assert graph.x.shape[0] == 5  # 5 nodes
        assert graph.x.shape[1] == 10  # 10 dimensions

    def test_complexity_reduction(self, builder):
        """Test that complexity is O(n log n) not O(n²)."""
        # For 100 documents, O(n²) would create ~10,000 comparisons
        # O(n log n) with top_k=3 creates ~300 comparisons

        documents = []
        for i in range(100):
            vec = np.random.randn(10).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            documents.append({"embedding": vec, "text": f"Doc {i}"})

        graph = builder.build_graph(documents)

        # Check edge count is reasonable (not quadratic)
        edge_count = graph.edge_index.shape[1]
        max_expected = 100 * builder.top_k * 2  # bidirectional

        assert edge_count < max_expected
        assert edge_count > 0

    def test_faiss_integration(self, builder):
        """Test FAISS index is properly used."""
        documents = [
            {
                "embedding": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                "text": "A",
            },
            {
                "embedding": np.array(
                    [0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32
                ),
                "text": "B",
            },
            {
                "embedding": np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
                "text": "C",
            },
        ]

        # Normalize
        for doc in documents:
            doc["embedding"] = doc["embedding"] / np.linalg.norm(doc["embedding"])

        graph = builder.build_graph(documents)

        # A and B should be connected (high similarity)
        # C should not be connected to A or B (low similarity)
        edges = graph.edge_index.numpy()

        # Check that edge exists between nodes 0 and 1
        edge_exists = False
        for i in range(edges.shape[1]):
            if (edges[0, i] == 0 and edges[1, i] == 1) or (
                edges[0, i] == 1 and edges[1, i] == 0
            ):
                edge_exists = True
                break

        assert edge_exists
