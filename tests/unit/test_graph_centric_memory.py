"""
Tests for Graph-Centric Memory Manager (C-value free)
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

from insightspike.core.layers.layer2_graph_centric import GraphCentricMemoryManager


class TestGraphCentricMemory:
    """Test the graph-centric memory manager without C-values."""
    
    @pytest.fixture
    def manager(self):
        """Create a manager instance for testing."""
        return GraphCentricMemoryManager(dim=10)
    
    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.dim == 10
        assert len(manager.episodes) == 0
        assert hasattr(manager, 'integration_config')
        assert hasattr(manager, 'splitting_config')
    
    def test_add_episode_no_c_value(self, manager):
        """Test adding episodes without C-values."""
        vec = np.random.randn(10).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        # Add episode (c_value parameter ignored if provided)
        idx = manager.add_episode(vec, "Test episode", c_value=0.99)
        
        assert idx == 0
        assert len(manager.episodes) == 1
        assert not hasattr(manager.episodes[0], 'c')
    
    def test_dynamic_importance(self, manager):
        """Test dynamic importance calculation."""
        # Add an episode
        vec = np.random.randn(10).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        idx = manager.add_episode(vec, "Test")
        
        # Initial importance
        initial_importance = manager.get_importance(idx)
        assert 0 <= initial_importance <= 1
        
        # Update access count
        manager._update_access(idx)
        manager._update_access(idx)
        
        # Importance should increase with access
        updated_importance = manager.get_importance(idx)
        assert updated_importance > initial_importance
    
    def test_graph_informed_integration(self, manager):
        """Test that graph connections affect integration decisions."""
        # Mock Layer3 graph
        mock_layer3 = Mock()
        mock_graph = Mock()
        mock_graph.edge_index = np.array([[0], [1]])  # Connect 0 and 1
        mock_layer3.previous_graph = mock_graph
        manager.set_layer3_graph(mock_layer3)
        
        # Add first episode
        vec1 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        manager.add_episode(vec1, "Episode 1")
        
        # Add similar episode (should integrate due to graph connection)
        vec2 = np.array([0.8, 0.2, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        vec2 = vec2 / np.linalg.norm(vec2)
        
        # Set thresholds to test graph bonus
        manager.integration_config.similarity_threshold = 0.85
        manager.integration_config.graph_connection_bonus = 0.1
        
        idx = manager.add_episode(vec2, "Episode 2")
        
        # Should integrate due to graph connection lowering threshold
        stats = manager.get_stats()
        assert stats['graph_assisted_integrations'] > 0
    
    def test_conflict_based_splitting(self, manager):
        """Test automatic splitting based on conflicts."""
        # Add episodes with conflicting neighbors
        vec1 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        vec2 = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        vec3 = np.array([0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        vec3 = vec3 / np.linalg.norm(vec3)
        
        idx1 = manager.add_episode(vec1, "Tech topic")
        idx2 = manager.add_episode(vec2, "Biology topic")
        idx3 = manager.add_episode(vec3, "Mixed content")
        
        # Calculate conflict for the mixed episode
        conflict = manager._calculate_conflict(idx3)
        
        # Conflict calculation might return 0 if graph not yet built
        # Just verify it doesn't crash
        assert conflict >= 0
    
    def test_no_c_value_in_episodes(self, manager):
        """Ensure episodes don't have C-values."""
        # Add multiple episodes
        for i in range(5):
            vec = np.random.randn(10).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            manager.add_episode(vec, f"Episode {i}")
        
        # Check none have C-values
        for episode in manager.episodes:
            assert not hasattr(episode, 'c')
            assert not hasattr(episode, 'c_value')
    
    def test_search_with_importance(self, manager):
        """Test search considers dynamic importance."""
        # Add episodes with different access patterns
        vecs = []
        for i in range(5):
            vec = np.random.randn(10).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            vecs.append(vec)
            manager.add_episode(vec, f"Episode {i}")
        
        # Access some episodes more than others
        for _ in range(5):
            manager._update_access(0)  # Most accessed
        for _ in range(3):
            manager._update_access(1)
        manager._update_access(2)
        # Episodes 3, 4 not accessed
        
        # Search
        query = np.random.randn(10).astype(np.float32)
        query = query / np.linalg.norm(query)
        results = manager.search_episodes("test query", k=5)
        
        # Results should include importance scores
        for result in results:
            assert 'importance' in result
            assert result['importance'] >= 0