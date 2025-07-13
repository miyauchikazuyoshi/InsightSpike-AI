"""
Unit tests for graph edge generation improvements
"""
import numpy as np
import pytest
from unittest.mock import Mock, patch
from dataclasses import dataclass

from insightspike.core.layers.scalable_graph_builder import ScalableGraphBuilder
from insightspike.core.learning.knowledge_graph_memory import KnowledgeGraphMemory


@dataclass
class MockReasoningConfig:
    similarity_threshold: float = 0.2


@dataclass
class MockScalableGraphConfig:
    top_k_neighbors: int = 50
    batch_size: int = 1000


@dataclass
class MockEmbeddingConfig:
    dimension: int = 384


@dataclass
class MockConfig:
    reasoning: MockReasoningConfig
    scalable_graph: MockScalableGraphConfig
    embedding: MockEmbeddingConfig

    def __init__(self, similarity_threshold=0.2, dimension=384, top_k=50):
        self.reasoning = MockReasoningConfig(similarity_threshold=similarity_threshold)
        self.scalable_graph = MockScalableGraphConfig(top_k_neighbors=top_k)
        self.embedding = MockEmbeddingConfig(dimension=dimension)


class TestGraphEdgeGeneration:
    """Test graph edge generation with improved thresholds."""

    def test_scalable_graph_builder_with_lower_threshold(self):
        """Test that ScalableGraphBuilder generates edges with lower threshold."""
        config = MockConfig(similarity_threshold=0.2, dimension=10, top_k=5)
        builder = ScalableGraphBuilder(config=config)

        # Create similar embeddings
        embeddings = []
        base_vec = np.random.randn(10).astype(np.float32)
        base_vec = base_vec / np.linalg.norm(base_vec)

        for i in range(10):
            # Add small noise to create similar vectors
            noise = np.random.randn(10).astype(np.float32) * 0.1
            vec = base_vec + noise
            vec = vec / np.linalg.norm(vec)
            embeddings.append(vec)

        # Build graph
        documents = [
            {"text": f"doc_{i}", "embedding": emb} for i, emb in enumerate(embeddings)
        ]
        graph = builder.build_graph(documents, embeddings=np.array(embeddings))

        # Assert edges were created
        assert graph.edge_index.shape[1] > 0, "No edges were created"
        assert graph.x.shape[0] == 10, "Wrong number of nodes"

    def test_knowledge_graph_memory_edge_creation(self):
        """Test that KnowledgeGraphMemory creates edges with threshold 0.2."""
        memory = KnowledgeGraphMemory(embedding_dim=10, similarity_threshold=0.2)

        # Create similar embeddings
        base_vec = np.random.randn(10).astype(np.float32)
        base_vec = base_vec / np.linalg.norm(base_vec)

        # Add nodes
        for i in range(5):
            noise = np.random.randn(10).astype(np.float32) * 0.1
            vec = base_vec + noise
            vec = vec / np.linalg.norm(vec)
            memory.add_episode_node(vec, i)

        # Check that edges were created
        assert memory.graph.edge_index.size(1) > 0, "No edges were created"
        assert memory.graph.x.size(0) == 5, "Wrong number of nodes"

    def test_edge_generation_with_diverse_embeddings(self):
        """Test edge generation with more diverse embeddings."""
        config = MockConfig(similarity_threshold=0.2, dimension=10, top_k=3)
        builder = ScalableGraphBuilder(config=config)

        # Create diverse embeddings
        embeddings = []
        for i in range(20):
            vec = np.random.randn(10).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            embeddings.append(vec)

        # Build graph
        documents = [
            {"text": f"doc_{i}", "embedding": emb} for i, emb in enumerate(embeddings)
        ]
        graph = builder.build_graph(documents, embeddings=np.array(embeddings))

        # Even with diverse embeddings, some edges should be created
        assert (
            graph.edge_index.shape[1] > 0
        ), "No edges were created even with lower threshold"
        assert graph.x.shape[0] == 20, "Wrong number of nodes"

    def test_similarity_threshold_effect(self):
        """Test the effect of different similarity thresholds."""
        embeddings = []
        base_vec = np.random.randn(10).astype(np.float32)
        base_vec = base_vec / np.linalg.norm(base_vec)

        # Create embeddings with varying similarity
        for i in range(10):
            noise = np.random.randn(10).astype(np.float32) * (0.1 + i * 0.05)
            vec = base_vec + noise
            vec = vec / np.linalg.norm(vec)
            embeddings.append(vec)

        embeddings = np.array(embeddings)
        documents = [
            {"text": f"doc_{i}", "embedding": emb} for i, emb in enumerate(embeddings)
        ]

        # Test with different thresholds
        edges_by_threshold = {}
        for threshold in [0.1, 0.2, 0.3, 0.4]:
            config = MockConfig(similarity_threshold=threshold, dimension=10, top_k=10)
            builder = ScalableGraphBuilder(config=config)
            graph = builder.build_graph(documents, embeddings=embeddings)
            edges_by_threshold[threshold] = graph.edge_index.shape[1]

        # Lower thresholds should create more edges
        assert edges_by_threshold[0.1] >= edges_by_threshold[0.2]
        assert edges_by_threshold[0.2] >= edges_by_threshold[0.3]
        assert edges_by_threshold[0.3] >= edges_by_threshold[0.4]

        # With 0.2 threshold, we should have a reasonable number of edges
        assert edges_by_threshold[0.2] > 0, "No edges with 0.2 threshold"

    def test_empty_graph_handling(self):
        """Test handling of empty graphs."""
        config = MockConfig(dimension=10)
        builder = ScalableGraphBuilder(config=config)

        # Build with empty documents
        graph = builder.build_graph([])

        assert graph.x.shape[0] == 0, "Empty graph should have no nodes"
        assert graph.edge_index.shape[1] == 0, "Empty graph should have no edges"

    def test_single_node_graph(self):
        """Test graph with single node."""
        config = MockConfig(dimension=10)
        builder = ScalableGraphBuilder(config=config)

        # Single document
        vec = np.random.randn(10).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        documents = [{"text": "single doc", "embedding": vec}]

        graph = builder.build_graph(documents, embeddings=np.array([vec]))

        assert graph.x.shape[0] == 1, "Single node graph should have 1 node"
        assert graph.edge_index.shape[1] == 0, "Single node graph should have no edges"
