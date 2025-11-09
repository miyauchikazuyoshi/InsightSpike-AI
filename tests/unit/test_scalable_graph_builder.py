"""
Test ScalableGraphBuilder
========================

Test the scalable graph building functionality.
"""

import numpy as np
import pytest

# Handle optional imports
try:
    import torch
    import pytest
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from insightspike.implementations.layers.scalable_graph_builder import (
    ScalableGraphBuilder,
)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestScalableGraphBuilder:
    """Test ScalableGraphBuilder functionality."""

    def test_init_with_config(self):
        """Test initialization with config."""
        # Create minimal config
        from types import SimpleNamespace
        
        config = SimpleNamespace(
            graph=SimpleNamespace(similarity_threshold=0.8),
            scalable_graph=SimpleNamespace(top_k_neighbors=10, batch_size=500),
            embedding=SimpleNamespace(dimension=128)
        )
        
        builder = ScalableGraphBuilder(config)
        assert builder.similarity_threshold == 0.8
        assert builder.top_k == 10
        assert builder.batch_size == 500
        assert builder.dimension == 128

    def test_init_without_config(self):
        """Test initialization without config uses defaults."""
        builder = ScalableGraphBuilder()
        assert builder.similarity_threshold == 0.7  # default
        assert builder.top_k == 50  # default
        assert builder.batch_size == 1000  # default
        assert builder.dimension == 384  # default

    def test_build_empty_graph(self):
        """Test building graph with no documents."""
        builder = ScalableGraphBuilder()
        graph = builder.build_graph([])
        
        assert isinstance(graph, Data)
        assert graph.num_nodes == 0
        assert graph.edge_index.size(1) == 0

    def test_build_single_node_graph(self):
        """Test building graph with single document."""
        builder = ScalableGraphBuilder()
        
        # Create document with embedding
        doc = {"text": "test", "embedding": np.random.randn(384)}
        graph = builder.build_graph([doc])
        
        assert graph.num_nodes == 1
        assert graph.x.shape == (1, 384)

    def test_build_multi_node_graph(self):
        """Test building graph with multiple documents."""
        builder = ScalableGraphBuilder()
        
        # Create documents with similar embeddings
        embedding1 = np.random.randn(384)
        embedding2 = embedding1 + np.random.randn(384) * 0.1  # Similar
        embedding3 = np.random.randn(384)  # Different
        
        docs = [
            {"text": "doc1", "embedding": embedding1},
            {"text": "doc2", "embedding": embedding2},
            {"text": "doc3", "embedding": embedding3},
        ]
        
        graph = builder.build_graph(docs)
        
        assert graph.num_nodes == 3
        assert graph.x.shape == (3, 384)
        # Should have edges between similar nodes
        assert graph.edge_index.size(1) > 0

    def test_incremental_update(self):
        """Test incremental graph updates."""
        builder = ScalableGraphBuilder()
        
        # Build initial graph
        docs1 = [
            {"text": f"doc{i}", "embedding": np.random.randn(384)}
            for i in range(3)
        ]
        graph1 = builder.build_graph(docs1)
        
        # Add more documents incrementally
        docs2 = [
            {"text": f"doc{i}", "embedding": np.random.randn(384)}
            for i in range(3, 5)
        ]
        graph2 = builder.build_graph(docs2, incremental=True)
        
        assert graph2.num_nodes == 5
        assert len(builder.documents) == 5

    def test_similarity_threshold(self):
        """Test that similarity threshold affects edge creation."""
        # Create two similar embeddings
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.9, 0.1, 0.0])  # cosine sim ~ 0.994
        
        docs = [
            {"embedding": emb1},
            {"embedding": emb2},
        ]
        
        # High threshold - should create edges
        builder_high = ScalableGraphBuilder()
        builder_high.similarity_threshold = 0.9
        graph_high = builder_high.build_graph(docs)
        
        # Very high threshold - no edges
        builder_vhigh = ScalableGraphBuilder()
        builder_vhigh.similarity_threshold = 0.999
        graph_vhigh = builder_vhigh.build_graph(docs)
        
        # High threshold should have edges, very high should not
        assert graph_high.edge_index.size(1) > 0
        assert graph_vhigh.edge_index.size(1) == 0

    def test_get_graph_stats(self):
        """Test getting graph statistics."""
        builder = ScalableGraphBuilder()
        
        # Before building
        stats = builder.get_graph_stats()
        assert stats["num_nodes"] == 0
        assert stats["num_edges"] == 0
        assert not stats["has_graph"]
        
        # After building
        docs = [
            {"embedding": np.random.randn(384)} for _ in range(5)
        ]
        builder.build_graph(docs)
        
        stats = builder.get_graph_stats()
        assert stats["num_nodes"] == 5
        assert stats["has_graph"]

    def test_missing_embeddings(self):
        """Test handling documents without embeddings."""
        builder = ScalableGraphBuilder()
        
        # Documents without embeddings should get random ones
        docs = [
            {"text": "doc1"},
            {"text": "doc2"},
        ]
        
        graph = builder.build_graph(docs)
        assert graph.num_nodes == 2
        assert graph.x.shape == (2, 384)

    def test_update_similarity_threshold(self):
        """Test updating similarity threshold."""
        builder = ScalableGraphBuilder()
        
        original = builder.similarity_threshold
        builder.update_similarity_threshold(0.5)
        assert builder.similarity_threshold == 0.5
        assert builder.similarity_threshold != original

    def test_build_alias_method(self):
        """Test the build() alias method."""
        builder = ScalableGraphBuilder()
        
        embeddings = np.random.randn(3, 384)
        graph = builder.build(embeddings)
        
        assert graph.num_nodes == 3
        assert graph.x.shape == (3, 384)

    @pytest.mark.parametrize("num_docs", [10, 50, 100])
    def test_scalability(self, num_docs):
        """Test building graphs of different sizes."""
        builder = ScalableGraphBuilder()
        builder.top_k = 5  # Limit connections for test speed
        
        docs = [
            {"embedding": np.random.randn(384)} for _ in range(num_docs)
        ]
        
        graph = builder.build_graph(docs)
        assert graph.num_nodes == num_docs
        # Each node should have at most top_k connections (bidirectional)
        assert graph.edge_index.size(1) <= num_docs * builder.top_k * 2
