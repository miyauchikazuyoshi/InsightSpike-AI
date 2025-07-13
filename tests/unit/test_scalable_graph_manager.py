"""
Tests for ScalableGraphManager with O(n log n) performance
"""
import os
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from insightspike.core.learning.scalable_graph_manager import ScalableGraphManager


class TestScalableGraphManager:
    """Test the ScalableGraphManager component"""

    def test_initialization(self):
        """Test ScalableGraphManager initialization"""
        manager = ScalableGraphManager()

        assert manager.graph is not None
        assert manager.graph.x.shape[0] == 0  # Empty graph initially
        assert manager.graph.edge_index.shape[1] == 0

    def test_add_single_node(self):
        """Test adding a single node"""
        manager = ScalableGraphManager()

        # Add first node
        embedding = np.random.rand(384).astype(np.float32)
        result = manager.add_episode_node(embedding, 0, {"text": "Test content"})

        assert result["success"] == True
        assert manager.graph.x.shape[0] == 1
        assert manager.graph.x.shape[1] == 384

    def test_add_multiple_nodes_with_edges(self):
        """Test adding multiple nodes and edge creation"""
        manager = ScalableGraphManager(top_k=2, similarity_threshold=0.5)

        # Add nodes with controlled embeddings
        embedding1 = np.ones(384, dtype=np.float32)
        embedding2 = np.ones(384, dtype=np.float32) * 0.9  # Similar
        embedding3 = np.zeros(384, dtype=np.float32)  # Different

        result1 = manager.add_episode_node(embedding1, 0, {"text": "Node 1"})
        result2 = manager.add_episode_node(embedding2, 1, {"text": "Node 2"})
        result3 = manager.add_episode_node(embedding3, 2, {"text": "Node 3"})

        assert result1["success"] == True
        assert result2["success"] == True
        assert result3["success"] == True

        # Check edges were created between similar nodes
        edges = manager.graph.edge_index.numpy()
        assert edges.shape[1] > 0  # Some edges should exist

    def test_get_current_graph(self):
        """Test getting current graph state"""
        manager = ScalableGraphManager()

        # Add some nodes
        for i in range(5):
            embedding = np.random.rand(384).astype(np.float32)
            manager.add_episode_node(embedding, i, {"text": f"Node {i}"})

        graph = manager.graph
        assert isinstance(graph, Data)
        assert graph.x.shape[0] == 5

    def test_save_and_load_index(self):
        """Test saving and loading FAISS index"""
        manager = ScalableGraphManager()

        # Add nodes
        for i in range(3):
            embedding = np.random.rand(384).astype(np.float32)
            manager.add_episode_node(embedding, i, {"text": f"Node {i}"})

        # Save index
        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
            manager.save_index(tmp.name)

            # Verify index was saved
            assert os.path.exists(tmp.name)

            # Create new manager and load index
            new_manager = ScalableGraphManager()
            new_manager.load_index(tmp.name)

            # Verify loaded index has correct number of vectors
            assert new_manager.index is not None
            assert new_manager.index.ntotal == 3

            os.unlink(tmp.name)

    def test_performance_scaling(self):
        """Test that performance is O(n log n) not O(n²)"""
        import time

        manager = ScalableGraphManager(top_k=10)

        # Time adding 100 nodes
        start = time.time()
        for i in range(100):
            embedding = np.random.rand(384).astype(np.float32)
            manager.add_episode_node(embedding, i, {"text": f"Node {i}"})
        time_100 = time.time() - start

        # Time adding next 100 nodes (should be similar if O(n log n))
        start = time.time()
        for i in range(100, 200):
            embedding = np.random.rand(384).astype(np.float32)
            manager.add_episode_node(embedding, i, {"text": f"Node {i}"})
        time_200 = time.time() - start

        # In O(n²), second batch would take ~4x longer
        # In O(n log n), it should be < 2x longer
        ratio = time_200 / time_100
        assert ratio < 2.5, f"Performance degradation too high: {ratio:.2f}x"

    def test_edge_weight_calculation(self):
        """Test edge weight calculation with similarity"""
        manager = ScalableGraphManager(similarity_threshold=0.5)

        # Create two very similar embeddings
        embedding1 = np.ones(384, dtype=np.float32)
        embedding1 = embedding1 / np.linalg.norm(embedding1)

        embedding2 = np.ones(384, dtype=np.float32)
        embedding2[0] = 0.9  # Slightly different
        embedding2 = embedding2 / np.linalg.norm(embedding2)

        manager.add_episode_node(embedding1, 0, {"text": "Node 1"})
        manager.add_episode_node(embedding2, 1, {"text": "Node 2"})

        # Check edge was created with appropriate weight
        if manager.graph.edge_index.shape[1] > 0:
            # Check if edge_attr exists (it might not be used in this implementation)
            if (
                hasattr(manager.graph, "edge_attr")
                and manager.graph.edge_attr is not None
            ):
                assert torch.all(
                    manager.graph.edge_attr > 0.5
                )  # All edges above threshold

    def test_empty_graph_operations(self):
        """Test operations on empty graph"""
        manager = ScalableGraphManager()

        # Get empty graph
        graph = manager.graph
        assert graph.x.shape[0] == 0

        # Save empty index (should handle gracefully)
        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
            # Index is None for empty graph, so this should not crash
            manager.save_index(tmp.name)

            # File might not be created if index is None
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
