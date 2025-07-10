"""
Comprehensive tests for ScalableGraphManager with FAISS integration.
"""

import pytest
import numpy as np
import torch
import faiss
from unittest.mock import patch, MagicMock
import tempfile
import os

from insightspike.core.learning.scalable_graph_manager import ScalableGraphManager


class TestScalableGraphManager:
    """Test suite for ScalableGraphManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a ScalableGraphManager instance."""
        return ScalableGraphManager(
            embedding_dim=384,
            similarity_threshold=0.3,
            top_k=50,
            conflict_threshold=0.8
        )
    
    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings."""
        np.random.seed(42)
        embeddings = []
        # Create clusters of similar embeddings
        for cluster in range(3):
            center = np.random.randn(384)
            for _ in range(5):
                noise = np.random.randn(384) * 0.1
                embedding = center + noise
                # Normalize
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
        return embeddings
    
    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.embedding_dim == 384
        assert manager.similarity_threshold == 0.3
        assert manager.top_k == 50
        assert manager.conflict_threshold == 0.8
        assert manager.index is None
        assert len(manager.embeddings) == 0
        assert manager.graph is not None
        assert manager.graph.num_nodes == 0
    
    def test_add_first_node(self, manager, sample_embeddings):
        """Test adding the first node."""
        result = manager.add_episode_node(
            embedding=sample_embeddings[0],
            index=0,
            metadata={"text": "First episode"}
        )
        
        assert result["success"]
        assert result["node_id"] == 0
        assert result["edges_added"] == 0
        assert len(result["conflicts"]) == 0
        assert manager.graph.num_nodes == 1
        assert manager.graph.x.size(0) == 1
    
    def test_add_multiple_nodes(self, manager, sample_embeddings):
        """Test adding multiple nodes and edge creation."""
        # Add nodes from same cluster (should create edges)
        for i in range(5):
            result = manager.add_episode_node(
                embedding=sample_embeddings[i],
                index=i,
                metadata={"text": f"Episode {i}"}
            )
            assert result["success"]
            assert result["node_id"] == i
        
        # Check graph structure
        assert manager.graph.num_nodes == 5
        assert manager.graph.edge_index.size(1) > 0  # Should have edges
        
        # Verify FAISS index
        assert manager.index.ntotal == 5
    
    def test_similarity_threshold(self, manager):
        """Test that edges are only created above similarity threshold."""
        # Create two very different embeddings
        emb1 = np.random.randn(384)
        emb1 = emb1 / np.linalg.norm(emb1)
        
        emb2 = -emb1  # Opposite direction
        
        manager.add_episode_node(emb1, 0)
        result = manager.add_episode_node(emb2, 1)
        
        assert result["success"]
        assert result["edges_added"] == 0  # Should not be connected
    
    def test_conflict_detection(self, manager):
        """Test conflict detection between nodes."""
        emb = np.random.randn(384)
        emb = emb / np.linalg.norm(emb)
        
        # Add two very similar nodes with conflicting content
        manager.add_episode_node(
            emb, 0,
            metadata={"text": "The model performance increased significantly"}
        )
        
        # Add nearly identical embedding with opposite meaning
        result = manager.add_episode_node(
            emb + np.random.randn(384) * 0.01,  # Very small noise
            1,
            metadata={"text": "The model performance decreased significantly"}
        )
        
        # Should detect conflict
        assert result["success"]
        assert len(result["conflicts"]) > 0
        assert result["conflicts"][0]["type"] == "directional"
    
    def test_node_importance_calculation(self, manager, sample_embeddings):
        """Test graph-based importance calculation."""
        # Create a hub node (connected to many others)
        hub_embedding = sample_embeddings[0]
        
        # Add hub
        manager.add_episode_node(hub_embedding, 0)
        
        # Add nodes similar to hub
        for i in range(1, 6):
            similar_emb = hub_embedding + np.random.randn(384) * 0.05
            similar_emb = similar_emb / np.linalg.norm(similar_emb)
            manager.add_episode_node(similar_emb, i)
        
        # Add isolated node
        isolated = -hub_embedding  # Very different
        result = manager.add_episode_node(isolated, 6)
        
        # Hub should have higher importance than isolated node
        hub_importance = manager._calculate_node_importance(0)
        isolated_importance = manager._calculate_node_importance(6)
        
        assert hub_importance > isolated_importance
    
    def test_get_subgraph(self, manager, sample_embeddings):
        """Test subgraph extraction."""
        # Add 10 nodes
        for i in range(10):
            manager.add_episode_node(sample_embeddings[i], i)
        
        # Get subgraph for first cluster (indices 0-4)
        subgraph = manager.get_subgraph([0, 1, 2, 3, 4])
        
        assert subgraph.num_nodes == 5
        assert subgraph.x.size(0) == 5
        # Should have edges within cluster
        assert subgraph.edge_index.size(1) > 0
    
    def test_should_split_episode(self, manager):
        """Test episode splitting logic."""
        # No conflicts - should not split
        assert not manager.should_split_episode([])
        
        # Single minor conflict - should not split
        minor_conflicts = [{
            "type": "temporal",
            "similarity": 0.7
        }]
        assert not manager.should_split_episode(minor_conflicts)
        
        # Multiple serious conflicts - should split
        serious_conflicts = [
            {"type": "directional", "similarity": 0.9},
            {"type": "directional", "similarity": 0.85}
        ]
        assert manager.should_split_episode(serious_conflicts)
    
    def test_get_split_candidates(self, manager):
        """Test finding split candidates from conflicts."""
        # Add conflict to history
        manager.conflict_history.append({
            "type": "directional",
            "nodes": [0, 5],
            "similarity": 0.9
        })
        manager.conflict_history.append({
            "type": "temporal",
            "nodes": [0, 3],
            "similarity": 0.85
        })
        
        candidates = manager.get_split_candidates(0)
        
        assert len(candidates) == 2
        assert candidates[0] == (5, 0.9)  # Highest similarity first
        assert candidates[1] == (3, 0.85)
    
    def test_update_from_episodes(self, manager, sample_embeddings):
        """Test rebuilding graph from episode list."""
        episodes = [
            {
                "embedding": emb,
                "text": f"Episode {i}",
                "timestamp": i * 1.0,
                "c_value": 0.5 + i * 0.1
            }
            for i, emb in enumerate(sample_embeddings[:5])
        ]
        
        result = manager.update_from_episodes(episodes)
        
        assert result["success"]
        assert result["graph_stats"]["nodes"] == 5
        assert result["graph_stats"]["edges"] > 0
        assert "density" in result["graph_stats"]
    
    def test_save_load_index(self, manager, sample_embeddings):
        """Test saving and loading FAISS index."""
        # Add some nodes
        for i in range(5):
            manager.add_episode_node(sample_embeddings[i], i)
        
        # Save index
        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
            manager.save_index(tmp.name)
            
            # Create new manager and load
            new_manager = ScalableGraphManager()
            new_manager.load_index(tmp.name)
            
            assert new_manager.index is not None
            assert new_manager.index.ntotal == 5
            
            # Cleanup
            os.unlink(tmp.name)
    
    def test_edge_case_empty_graph(self, manager):
        """Test operations on empty graph."""
        # Get subgraph from empty graph
        subgraph = manager.get_subgraph([])
        # get_subgraph returns None for empty input
        assert subgraph is None or subgraph.num_nodes == 0
        
        # Get current graph
        graph = manager.get_current_graph()
        assert graph.num_nodes == 0
        
        # Calculate importance for non-existent node
        importance = manager._calculate_node_importance(0)
        assert importance == 0.0
    
    def test_temporal_conflict_detection(self, manager):
        """Test temporal conflict detection."""
        emb = np.random.randn(384)
        emb = emb / np.linalg.norm(emb)
        
        # Add two nodes very close in time
        manager.add_episode_node(
            emb, 0,
            metadata={"text": "First observation", "timestamp": 1.0}
        )
        
        result = manager.add_episode_node(
            emb + np.random.randn(384) * 0.01, 1,
            metadata={"text": "Different observation", "timestamp": 1.5}
        )
        
        # Should detect temporal conflict
        conflicts = result.get("conflicts", [])
        temporal_conflicts = [c for c in conflicts if c["type"] == "temporal"]
        assert len(temporal_conflicts) > 0
    
    def test_performance_with_many_nodes(self, manager):
        """Test performance with larger graph."""
        # Add 100 nodes
        for i in range(100):
            emb = np.random.randn(384)
            emb = emb / np.linalg.norm(emb)
            
            result = manager.add_episode_node(emb, i)
            assert result["success"]
            
            # Build time should remain reasonable
            if i > 50:
                assert result["build_time"] < 1.0  # Less than 1 second
    
    def test_error_handling(self, manager):
        """Test error handling for invalid inputs."""
        # Invalid embedding dimension
        wrong_dim_emb = np.random.randn(256)
        result = manager.add_episode_node(wrong_dim_emb, 0)
        assert not result["success"]
        assert "error" in result
        
        # None embedding
        result = manager.add_episode_node(None, 0)
        assert not result["success"]
    
    def test_normalized_embeddings(self, manager):
        """Test that embeddings are properly normalized."""
        # Add unnormalized embedding
        emb = np.random.randn(384) * 10  # Large values
        manager.add_episode_node(emb, 0)
        
        # Check stored embedding is normalized
        stored_emb = manager.embeddings[0]
        norm = np.linalg.norm(stored_emb)
        assert abs(norm - 1.0) < 1e-6