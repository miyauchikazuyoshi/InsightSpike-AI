"""
Tests for ScalableGraphManager with O(n log n) performance
"""
import pytest
import torch
import numpy as np
from torch_geometric.data import Data
from unittest.mock import Mock, patch
import tempfile
import os

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
        node_idx = manager.add_node(embedding, {"text": "Test content"})
        
        assert node_idx == 0
        assert manager.graph.x.shape[0] == 1
        assert manager.graph.x.shape[1] == 384
        
    def test_add_multiple_nodes_with_edges(self):
        """Test adding multiple nodes and edge creation"""
        manager = ScalableGraphManager(top_k=2, similarity_threshold=0.5)
        
        # Add nodes with controlled embeddings
        embedding1 = np.ones(384, dtype=np.float32)
        embedding2 = np.ones(384, dtype=np.float32) * 0.9  # Similar
        embedding3 = np.zeros(384, dtype=np.float32)  # Different
        
        idx1 = manager.add_node(embedding1, {"text": "Node 1"})
        idx2 = manager.add_node(embedding2, {"text": "Node 2"})
        idx3 = manager.add_node(embedding3, {"text": "Node 3"})
        
        assert idx1 == 0
        assert idx2 == 1
        assert idx3 == 2
        
        # Check edges were created between similar nodes
        edges = manager.graph.edge_index.numpy()
        assert edges.shape[1] > 0  # Some edges should exist
        
    def test_get_current_graph(self):
        """Test getting current graph state"""
        manager = ScalableGraphManager()
        
        # Add some nodes
        for i in range(5):
            embedding = np.random.rand(384).astype(np.float32)
            manager.add_node(embedding, {"text": f"Node {i}"})
        
        graph = manager.get_current_graph()
        assert isinstance(graph, Data)
        assert graph.x.shape[0] == 5
        
    def test_save_and_load_graph(self):
        """Test saving and loading graph"""
        manager = ScalableGraphManager()
        
        # Add nodes
        for i in range(3):
            embedding = np.random.rand(384).astype(np.float32)
            manager.add_node(embedding, {"text": f"Node {i}"})
        
        # Save graph
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            manager.save_graph(tmp.name)
            
            # Create new manager and load
            new_manager = ScalableGraphManager()
            new_manager.load_graph(tmp.name)
            
            # Verify loaded graph
            assert new_manager.graph.x.shape[0] == 3
            assert torch.allclose(new_manager.graph.x, manager.graph.x)
            
            os.unlink(tmp.name)
    
    def test_performance_scaling(self):
        """Test that performance is O(n log n) not O(n²)"""
        import time
        
        manager = ScalableGraphManager(top_k=10)
        
        # Time adding 100 nodes
        start = time.time()
        for i in range(100):
            embedding = np.random.rand(384).astype(np.float32)
            manager.add_node(embedding, {"text": f"Node {i}"})
        time_100 = time.time() - start
        
        # Time adding next 100 nodes (should be similar if O(n log n))
        start = time.time()
        for i in range(100, 200):
            embedding = np.random.rand(384).astype(np.float32)
            manager.add_node(embedding, {"text": f"Node {i}"})
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
        
        manager.add_node(embedding1, {"text": "Node 1"})
        manager.add_node(embedding2, {"text": "Node 2"})
        
        # Check edge was created with appropriate weight
        if manager.graph.edge_index.shape[1] > 0:
            edge_attr = manager.graph.edge_attr
            assert edge_attr is not None
            assert torch.all(edge_attr > 0.5)  # All edges above threshold
            
    def test_empty_graph_operations(self):
        """Test operations on empty graph"""
        manager = ScalableGraphManager()
        
        # Get empty graph
        graph = manager.get_current_graph()
        assert graph.x.shape[0] == 0
        
        # Save empty graph
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            manager.save_graph(tmp.name)
            assert os.path.exists(tmp.name)
            os.unlink(tmp.name)