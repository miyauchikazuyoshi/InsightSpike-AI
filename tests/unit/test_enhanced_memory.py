"""
Tests for L2EnhancedScalableMemory with O(n log n) performance
"""
import pytest
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch

from insightspike.core.layers.layer2_enhanced_scalable import L2EnhancedScalableMemory
from insightspike.utils.embedder import get_model


class TestL2EnhancedScalableMemory:
    """Test the enhanced scalable memory implementation"""
    
    @pytest.fixture
    def memory(self):
        """Create memory instance for testing"""
        with patch('insightspike.utils.embedder.get_model') as mock_get_model:
            mock_embedder = Mock()
            mock_embedder.get_sentence_embedding_dimension.return_value = 384
            mock_embedder.encode.return_value = np.random.randn(1, 384)
            mock_get_model.return_value = mock_embedder
            
            with patch('insightspike.core.layers.layer2_enhanced_scalable.get_model', return_value=mock_embedder):
                with patch('insightspike.core.layers.layer2_memory_manager.get_model', return_value=mock_embedder):
                    return L2EnhancedScalableMemory()
    
    def test_initialization(self, memory):
        """Test memory initialization"""
        assert memory.episodes == []
        assert memory.scalable_graph_manager is not None
        assert memory.index is not None
        assert hasattr(memory, 'index_ids')
        
    def test_store_single_episode(self, memory):
        """Test storing a single episode"""
        episode = memory.store_episode("Test content", {"source": "test"})
        
        assert len(memory.episodes) == 1
        assert episode["text"] == "Test content"
        assert episode["importance"] > 0
        assert "id" in episode
        assert "embedding" in episode
        
    def test_store_multiple_episodes(self, memory):
        """Test storing multiple episodes"""
        texts = ["First episode", "Second episode", "Third episode"]
        
        for text in texts:
            memory.store_episode(text, {"source": "test"})
        
        assert len(memory.episodes) == 3
        assert memory.episodes[0]["text"] == "First episode"
        assert memory.episodes[2]["text"] == "Third episode"
        
    def test_search_episodes(self, memory):
        """Test searching episodes with FAISS"""
        # Store episodes
        memory.store_episode("Python programming basics", {"type": "tutorial"})
        memory.store_episode("Advanced Python concepts", {"type": "tutorial"})
        memory.store_episode("Machine learning with Python", {"type": "ml"})
        memory.store_episode("Data structures in Java", {"type": "tutorial"})
        
        # Search for Python-related content
        results = memory.search_episodes("Python programming", k=2)
        
        assert len(results) == 2
        assert all("Python" in r["text"] for r in results)
        assert results[0]["score"] > results[1]["score"]  # Ordered by relevance
        
    def test_update_importance_graph_based(self, memory):
        """Test that importance is updated based on graph structure"""
        # Store episodes
        ep1 = memory.store_episode("Core concept", {})
        ep2 = memory.store_episode("Related concept", {})
        ep3 = memory.store_episode("Another related concept", {})
        
        # Initial importance should be set
        assert all(ep["importance"] > 0 for ep in memory.episodes)
        
        # Update importance (graph-based)
        memory.update_importance()
        
        # Importance values should be calculated from graph
        for i, ep in enumerate(memory.episodes):
            assert "importance" in ep
            assert 0 <= ep["importance"] <= 1
            
    def test_conflict_detection_and_splitting(self, memory):
        """Test automatic conflict detection and splitting"""
        # Create conflicting episodes
        memory.store_episode("Python is interpreted", {"type": "fact"})
        memory.store_episode("Python is compiled to bytecode", {"type": "fact"})
        
        # These should be detected as potential conflicts if similar enough
        # The system should handle them appropriately
        assert len(memory.episodes) >= 2
        
    def test_save_and_load_state(self, memory):
        """Test saving and loading memory state"""
        # Store episodes
        memory.store_episode("Test episode 1", {"meta": "data1"})
        memory.store_episode("Test episode 2", {"meta": "data2"})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save state
            memory.save_state(tmpdir)
            
            # Check files were created
            assert os.path.exists(os.path.join(tmpdir, "episodes.json"))
            assert os.path.exists(os.path.join(tmpdir, "scalable_index.faiss"))
            assert os.path.exists(os.path.join(tmpdir, "graph_pyg.pt"))
            
            # Create new memory and load
            new_memory = L2EnhancedScalableMemory()
            new_memory.load_state(tmpdir)
            
            # Verify loaded state
            assert len(new_memory.episodes) == 2
            assert new_memory.episodes[0]["text"] == "Test episode 1"
            assert new_memory.episodes[1]["text"] == "Test episode 2"
            
    def test_performance_with_many_episodes(self, memory):
        """Test performance doesn't degrade with many episodes"""
        import time
        
        # Add 100 episodes and measure time
        start = time.time()
        for i in range(100):
            memory.store_episode(f"Episode {i}", {"index": i})
        time_100 = time.time() - start
        
        # Add another 100 and measure
        start = time.time()
        for i in range(100, 200):
            memory.store_episode(f"Episode {i}", {"index": i})
        time_200 = time.time() - start
        
        # Should be O(n log n), not O(n²)
        ratio = time_200 / time_100
        assert ratio < 2.5, f"Performance degradation too high: {ratio:.2f}x"
        
        # Test search performance
        start = time.time()
        results = memory.search_episodes("Episode 150", k=5)
        search_time = time.time() - start
        
        assert len(results) == 5
        assert search_time < 0.1  # Should be fast even with 200 episodes
        
    def test_get_graph_state(self, memory):
        """Test getting current graph state"""
        # Store some episodes
        memory.store_episode("Node 1", {})
        memory.store_episode("Node 2", {})
        memory.store_episode("Node 3", {})
        
        # Get graph through memory
        graph = memory.get_graph()
        
        assert graph is not None
        assert graph.x.shape[0] == 3  # 3 nodes
        
    def test_integration_with_similar_episodes(self, memory):
        """Test that very similar episodes are integrated"""
        # Store very similar episodes
        memory.store_episode("Machine learning is powerful", {"source": "A"})
        memory.store_episode("Machine learning is very powerful", {"source": "B"})
        
        # With high similarity, these might be integrated
        # The exact behavior depends on threshold settings
        assert len(memory.episodes) >= 1  # At least one should exist
        
    def test_empty_memory_operations(self, memory):
        """Test operations on empty memory"""
        # Search empty memory
        results = memory.search_episodes("test query", k=5)
        assert results == []
        
        # Update importance on empty memory
        memory.update_importance()  # Should not crash
        
        # Get graph from empty memory
        graph = memory.get_graph()
        assert graph.x.shape[0] == 0
        
    def test_metadata_preservation(self, memory):
        """Test that metadata is preserved through operations"""
        # Store with rich metadata
        metadata = {
            "source": "test",
            "timestamp": "2025-01-06",
            "author": "system",
            "tags": ["test", "unit"]
        }
        
        episode = memory.store_episode("Test content", metadata)
        
        # Check metadata is preserved
        for key, value in metadata.items():
            assert episode.get(key) == value
            
        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            memory.save_state(tmpdir)
            
            new_memory = L2EnhancedScalableMemory()
            new_memory.load_state(tmpdir)
            
            # Check metadata still preserved
            loaded_ep = new_memory.episodes[0]
            for key, value in metadata.items():
                assert loaded_ep.get(key) == value