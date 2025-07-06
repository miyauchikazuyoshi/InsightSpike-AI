"""
Unit tests for scalable graph features
=====================================
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile

from insightspike.core.layers.scalable_graph_builder import ScalableGraphBuilder
from insightspike.core.learning.scalable_graph_manager import ScalableGraphManager
from insightspike.utils.graph_importance import GraphImportanceCalculator
from insightspike.monitoring import GraphOperationMonitor
from insightspike.core.config import get_config


class TestScalableGraphBuilder:
    """Test the ScalableGraphBuilder with FAISS."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = get_config()
        self.builder = ScalableGraphBuilder(self.config)
        
        # Create test documents
        self.test_docs = [
            {"text": f"Document {i}", "embedding": np.random.randn(384).astype(np.float32)}
            for i in range(100)
        ]
        
        # Normalize embeddings
        for doc in self.test_docs:
            doc["embedding"] = doc["embedding"] / np.linalg.norm(doc["embedding"])
    
    def test_build_graph_from_scratch(self):
        """Test building a graph from scratch."""
        graph = self.builder.build_graph(self.test_docs[:10])
        
        assert graph is not None
        assert graph.num_nodes == 10
        assert graph.x.shape == (10, 384)
        assert graph.edge_index.shape[0] == 2
        
        # Should have some edges (but not fully connected)
        num_edges = graph.edge_index.shape[1]
        assert num_edges > 0
        assert num_edges < 90  # Less than fully connected (10*9)
    
    def test_incremental_update(self):
        """Test incremental graph updates."""
        # Build initial graph
        graph1 = self.builder.build_graph(self.test_docs[:10])
        initial_nodes = graph1.num_nodes
        
        # Add more documents incrementally
        graph2 = self.builder.build_graph(
            self.test_docs[10:20],
            incremental=True
        )
        
        assert graph2.num_nodes == 20
        assert graph2.num_nodes > initial_nodes
        assert graph2.edge_index.shape[1] > graph1.edge_index.shape[1]
    
    def test_similarity_threshold(self):
        """Test that similarity threshold affects edge creation."""
        # Low threshold - more edges
        self.builder.similarity_threshold = 0.1
        graph_low = self.builder.build_graph(self.test_docs[:10])
        edges_low = graph_low.edge_index.shape[1]
        
        # High threshold - fewer edges
        self.builder.similarity_threshold = 0.8
        self.builder.index = None  # Reset index
        graph_high = self.builder.build_graph(self.test_docs[:10])
        edges_high = graph_high.edge_index.shape[1]
        
        assert edges_high < edges_low
    
    def test_get_neighbors(self):
        """Test neighbor retrieval."""
        self.builder.build_graph(self.test_docs[:20])
        
        # Get neighbors for node 0
        distances, neighbors = self.builder.get_neighbors(0, k=5)
        
        assert len(distances) == 5
        assert len(neighbors) == 5
        assert all(n != 0 for n in neighbors)  # No self-connections
        assert all(d >= 0 for d in distances)  # Valid distances
    
    def test_monitoring_integration(self):
        """Test integration with monitoring."""
        monitor = GraphOperationMonitor(enable_file_logging=False)
        builder = ScalableGraphBuilder(self.config, monitor)
        
        # Build graph with monitoring
        graph = builder.build_graph(self.test_docs[:10])
        
        # Check that operations were recorded
        summary = monitor.get_operation_summary("build_graph")
        assert summary["count"] == 1
        assert summary["avg_duration"] > 0
        assert summary["avg_nodes_added"] == 10


class TestScalableGraphManager:
    """Test the ScalableGraphManager."""
    
    def setup_method(self):
        """Setup test environment."""
        self.manager = ScalableGraphManager()
        
        # Create test embeddings
        self.test_embeddings = [
            np.random.randn(384).astype(np.float32)
            for _ in range(50)
        ]
        
        # Normalize
        for i, emb in enumerate(self.test_embeddings):
            self.test_embeddings[i] = emb / np.linalg.norm(emb)
    
    def test_add_episode_node(self):
        """Test adding episode nodes."""
        # Add first node
        result1 = self.manager.add_episode_node(
            self.test_embeddings[0],
            0,
            {"text": "First episode"}
        )
        
        assert result1["success"]
        assert result1["node_id"] == 0
        assert result1["edges_added"] == 0  # No edges for first node
        
        # Add second node
        result2 = self.manager.add_episode_node(
            self.test_embeddings[1],
            1,
            {"text": "Second episode"}
        )
        
        assert result2["success"]
        assert result2["node_id"] == 1
        # May or may not have edges depending on similarity
    
    def test_conflict_detection(self):
        """Test conflict detection between nodes."""
        # Create conflicting texts
        meta1 = {"text": "The stock price will increase significantly"}
        meta2 = {"text": "The stock price will decrease significantly"}
        
        # Use very similar embeddings to trigger conflict detection
        emb1 = self.test_embeddings[0]
        emb2 = emb1 + np.random.randn(384) * 0.01  # Small perturbation
        emb2 = emb2 / np.linalg.norm(emb2)
        
        # Set high conflict threshold to trigger
        self.manager.conflict_threshold = 0.7
        
        # Add nodes
        self.manager.add_episode_node(emb1, 0, meta1)
        result = self.manager.add_episode_node(emb2, 1, meta2)
        
        # Should detect conflict
        assert len(result.get("conflicts", [])) > 0
    
    def test_importance_calculation(self):
        """Test node importance calculation."""
        # Build a small graph
        for i in range(10):
            self.manager.add_episode_node(
                self.test_embeddings[i],
                i,
                {"text": f"Episode {i}"}
            )
        
        # Calculate importance for central node
        importance = self.manager._calculate_node_importance(5)
        
        assert isinstance(importance, float)
        assert 0 <= importance <= 1
    
    def test_should_split_episode(self):
        """Test episode splitting logic."""
        # No conflicts - should not split
        assert not self.manager.should_split_episode([])
        
        # One conflict - should not split
        conflicts = [{"type": "directional", "similarity": 0.9}]
        assert not self.manager.should_split_episode(conflicts)
        
        # Multiple serious conflicts - should split
        conflicts = [
            {"type": "directional", "similarity": 0.95},
            {"type": "directional", "similarity": 0.92}
        ]
        assert self.manager.should_split_episode(conflicts)
    
    def test_save_load_index(self):
        """Test saving and loading FAISS index."""
        # Build graph
        for i in range(10):
            self.manager.add_episode_node(
                self.test_embeddings[i],
                i,
                {"text": f"Episode {i}"}
            )
        
        # Save index
        with tempfile.NamedTemporaryFile(suffix=".faiss") as tmp:
            self.manager.save_index(tmp.name)
            
            # Create new manager and load
            new_manager = ScalableGraphManager()
            new_manager.load_index(tmp.name)
            
            # Verify index is loaded
            assert new_manager.index is not None


class TestGraphImportanceCalculator:
    """Test graph-based importance calculation."""
    
    def setup_method(self):
        """Setup test environment."""
        self.calculator = GraphImportanceCalculator()
        
        # Create test graph
        x = torch.randn(10, 384)
        # Create a simple connected graph
        edges = []
        for i in range(9):
            edges.extend([[i, i+1], [i+1, i]])  # Bidirectional
        
        self.graph = torch_geometric.data.Data(
            x=x,
            edge_index=torch.tensor(edges, dtype=torch.long).t()
        )
        self.graph.num_nodes = 10
    
    def test_degree_centrality(self):
        """Test degree centrality calculation."""
        # Node 0 and 9 have degree 1 (endpoints)
        # Nodes 1-8 have degree 2
        
        importance_0 = self.calculator._degree_centrality(self.graph, 0)
        importance_5 = self.calculator._degree_centrality(self.graph, 5)
        
        assert importance_0 < importance_5  # Endpoint vs middle node
        assert 0 <= importance_0 <= 1
        assert 0 <= importance_5 <= 1
    
    def test_pagerank_calculation(self):
        """Test PageRank calculation."""
        pagerank = self.calculator._calculate_pagerank(self.graph)
        
        assert len(pagerank) == 10
        assert all(0 <= score <= 1 for score in pagerank)
        assert abs(pagerank.sum() - 1.0) < 0.01  # Should sum to 1
    
    def test_access_tracking(self):
        """Test access-based importance."""
        # Access node 5 multiple times
        for _ in range(5):
            self.calculator._update_access(5)
        
        # Access node 0 once
        self.calculator._update_access(0)
        
        # Compare access scores
        score_0 = self.calculator._access_score(0)
        score_5 = self.calculator._access_score(5)
        
        assert score_5 > score_0  # More accesses = higher score
    
    def test_combined_importance(self):
        """Test combined importance calculation."""
        # Calculate importance for a node
        scores = self.calculator.calculate_importance(self.graph, 5)
        
        assert "degree" in scores
        assert "betweenness" in scores
        assert "pagerank" in scores
        assert "access" in scores
        assert "combined" in scores
        
        # Combined should be weighted average
        assert 0 <= scores["combined"] <= 1
    
    def test_cache_invalidation(self):
        """Test that cache is properly invalidated."""
        # Calculate importance (fills cache)
        scores1 = self.calculator.calculate_importance(self.graph, 5)
        
        # Invalidate cache
        self.calculator.invalidate_cache()
        
        # Calculate again
        scores2 = self.calculator.calculate_importance(self.graph, 5)
        
        # Should get same results (deterministic)
        assert abs(scores1["pagerank"] - scores2["pagerank"]) < 0.01


# Import torch_geometric for test
try:
    import torch_geometric
except ImportError:
    torch_geometric = None


@pytest.mark.skipif(
    torch_geometric is None,
    reason="torch_geometric not available"
)
def test_integration():
    """Test integration of all scalable components."""
    config = get_config()
    monitor = GraphOperationMonitor(enable_file_logging=False)
    
    # Create components
    builder = ScalableGraphBuilder(config, monitor)
    manager = ScalableGraphManager()
    calculator = GraphImportanceCalculator()
    
    # Create test documents
    docs = [
        {"text": f"Test document {i}", "embedding": np.random.randn(384)}
        for i in range(20)
    ]
    
    # Normalize embeddings
    for doc in docs:
        doc["embedding"] = doc["embedding"] / np.linalg.norm(doc["embedding"])
    
    # Build graph
    graph = builder.build_graph(docs)
    
    # Add to manager
    for i, doc in enumerate(docs):
        manager.add_episode_node(
            doc["embedding"],
            i,
            {"text": doc["text"]}
        )
    
    # Calculate importance
    importance_map = calculator.update_graph_importance(graph)
    
    # Verify integration
    assert graph.num_nodes == 20
    assert len(importance_map) == 20
    assert all(0 <= score <= 1 for score in importance_map.values())
    
    # Check monitoring
    summary = monitor.get_operation_summary()
    assert "build_graph" in summary
    assert summary["build_graph"]["count"] == 1