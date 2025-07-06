"""
Comprehensive tests for L2EnhancedScalableMemory with conflict-based splitting.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock
import tempfile
from pathlib import Path

from src.insightspike.core.layers.layer2_enhanced_scalable import L2EnhancedScalableMemory
from src.insightspike.core.layers.layer2_memory_manager import Episode


class TestL2EnhancedScalableMemory:
    """Test suite for L2EnhancedScalableMemory."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock()
        config.memory.embedding_dim = 384
        config.memory.index_file = "data/test_index.faiss"
        config.memory.c_min = 0.1
        config.memory.c_max = 1.0
        config.memory.c_decay_rate = 0.01
        config.memory.storage_threshold = 0.5
        config.reasoning.similarity_threshold = 0.3
        config.reasoning.graph_top_k = 50
        config.reasoning.conflict_threshold = 0.8
        return config
    
    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        embedder = MagicMock()
        embedder.get_sentence_embedding_dimension.return_value = 384
        
        def encode_side_effect(texts, **kwargs):
            # Return normalized random embeddings
            embeddings = []
            for _ in texts:
                emb = np.random.randn(384)
                emb = emb / np.linalg.norm(emb)
                embeddings.append(emb)
            return np.array(embeddings)
        
        embedder.encode.side_effect = encode_side_effect
        return embedder
    
    @pytest.fixture
    def memory(self, mock_config, mock_embedder):
        """Create L2EnhancedScalableMemory instance."""
        with patch('src.insightspike.core.layers.layer2_enhanced_scalable.get_config', return_value=mock_config):
            with patch('src.insightspike.core.layers.layer2_enhanced_scalable.get_model', return_value=mock_embedder):
                with patch('src.insightspike.core.layers.layer2_memory_manager.get_model', return_value=mock_embedder):
                    return L2EnhancedScalableMemory(
                        dim=384,
                        config=mock_config,
                        use_scalable_graph=True,
                        conflict_split_threshold=2
                    )
    
    def test_initialization(self, memory):
        """Test memory initialization."""
        assert memory.use_scalable_graph
        assert memory.conflict_split_threshold == 2
        assert memory.scalable_graph is not None
        assert len(memory.recent_conflicts) == 0
    
    def test_store_episode_with_graph(self, memory, mock_embedder):
        """Test storing episode with graph operations."""
        result = memory.store_episode("Test episode content", c_value=0.7)
        
        assert result
        assert len(memory.episodes) == 1
        assert memory.scalable_graph.graph.num_nodes == 1
    
    def test_conflict_detection_and_splitting(self, memory, mock_embedder):
        """Test conflict detection triggers splitting."""
        # Create mock conflict in scalable graph
        with patch.object(memory.scalable_graph, 'add_episode_node') as mock_add:
            mock_add.return_value = {
                "success": True,
                "conflicts": [
                    {"type": "directional", "nodes": [0, 1], "similarity": 0.9},
                    {"type": "directional", "nodes": [0, 2], "similarity": 0.85}
                ],
                "importance": 0.5
            }
            
            with patch.object(memory.scalable_graph, 'should_split_episode', return_value=True):
                with patch.object(memory, '_handle_conflict_split') as mock_split:
                    result = memory.store_episode("Conflicting content", c_value=0.6)
                    
                    assert result
                    mock_split.assert_called_once()
    
    def test_analyze_text_for_splits(self, memory):
        """Test text analysis for splitting."""
        # Single sentence - should not split
        text1 = "This is a single sentence"
        parts1 = memory._analyze_text_for_splits(text1)
        assert len(parts1) == 1
        assert parts1[0] == text1
        
        # Multiple sentences - should split
        text2 = "First sentence. Second sentence! Third sentence? Fourth sentence."
        parts2 = memory._analyze_text_for_splits(text2)
        assert len(parts2) > 1
        
        # Very short parts - should be combined
        text3 = "A. B. C. This is a longer sentence that should be separate."
        parts3 = memory._analyze_text_for_splits(text3)
        assert len(parts3) == 2  # Short parts combined
    
    def test_split_by_conflict(self, memory, mock_embedder):
        """Test episode splitting by conflict."""
        # Add initial episode
        memory.store_episode("This is the first part. This is the second part.", c_value=0.8)
        
        # Perform split
        conflict_candidates = [(0, 0.9)]
        new_indices = memory.split_by_conflict(0, conflict_candidates)
        
        assert len(new_indices) == 2  # Split into 2 parts
        assert len(memory.episodes) == 2  # Original removed, 2 added
        
        # Check metadata
        for idx in new_indices:
            episode = memory.episodes[idx]
            assert "split_from" in episode.metadata
            assert episode.metadata["split_reason"] == "conflict"
    
    def test_update_episode_importance(self, memory):
        """Test updating episode importance from graph."""
        memory.store_episode("Test episode", c_value=0.5)
        
        # Update importance
        memory._update_episode_importance(0, 0.8)
        
        # Check C-value was updated with blending
        episode = memory.episodes[0]
        # Should be blend of 0.5 and 0.8
        assert 0.5 < episode.c < 0.8
    
    def test_search_episodes_with_graph(self, memory, mock_embedder):
        """Test enhanced search with graph reranking."""
        # Add multiple episodes
        for i in range(5):
            memory.store_episode(f"Episode {i} content", c_value=0.5 + i * 0.1)
        
        # Mock graph importance calculation
        with patch.object(memory.scalable_graph, '_calculate_node_importance') as mock_importance:
            mock_importance.side_effect = lambda idx: 0.1 * idx  # Higher index = higher importance
            
            results = memory.search_episodes_with_graph("test query", k=3)
            
            assert len(results) <= 3
            for result in results:
                assert "graph_importance" in result
                assert "enhanced_score" in result
    
    def test_get_graph_stats(self, memory):
        """Test getting graph statistics."""
        # Add some episodes
        for i in range(3):
            memory.store_episode(f"Episode {i}", c_value=0.5)
        
        stats = memory.get_graph_stats()
        
        assert stats["graph_enabled"]
        assert stats["nodes"] == 3
        assert "edges" in stats
        assert "density" in stats
        assert "recent_conflicts" in stats
        assert stats["faiss_index_size"] == 3
    
    def test_save_and_load_with_graph(self, memory, mock_embedder, mock_config):
        """Test saving and loading with scalable graph."""
        # Add episodes
        for i in range(3):
            memory.store_episode(f"Episode {i}", c_value=0.5 + i * 0.1)
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_index.faiss"
            
            with patch('src.insightspike.core.layers.layer2_enhanced_scalable.get_config', return_value=mock_config):
                result = memory.save(save_path)
                assert result
                
                # Check scalable index was saved
                scalable_path = save_path.parent / "scalable_index.faiss"
                assert scalable_path.exists()
                
                # Create new memory and load
                with patch('src.insightspike.core.layers.layer2_memory_manager.get_model', return_value=mock_embedder):
                    new_memory = L2EnhancedScalableMemory(
                        dim=384,
                        config=mock_config,
                        use_scalable_graph=True
                    )
                    
                    result = new_memory.load(save_path)
                    assert result
                    assert len(new_memory.episodes) == 3
                    assert new_memory.scalable_graph.graph.num_nodes == 3
    
    def test_disabled_scalable_graph(self, mock_config, mock_embedder):
        """Test memory with scalable graph disabled."""
        with patch('src.insightspike.core.layers.layer2_enhanced_scalable.get_config', return_value=mock_config):
            with patch('src.insightspike.core.layers.layer2_enhanced_scalable.get_model', return_value=mock_embedder):
                with patch('src.insightspike.core.layers.layer2_memory_manager.get_model', return_value=mock_embedder):
                    memory = L2EnhancedScalableMemory(
                        dim=384,
                        config=mock_config,
                        use_scalable_graph=False
                    )
        
        assert not memory.use_scalable_graph
        assert memory.scalable_graph is None
        
        # Should still work without graph
        result = memory.store_episode("Test without graph", c_value=0.5)
        assert result
        
        stats = memory.get_graph_stats()
        assert not stats["graph_enabled"]
    
    def test_handle_conflict_split_with_candidates(self, memory, mock_embedder):
        """Test handling conflict split with multiple candidates."""
        # Add initial episodes
        for i in range(3):
            memory.store_episode(f"Episode {i} content", c_value=0.5)
        
        # Mock get_split_candidates to return multiple candidates
        with patch.object(memory.scalable_graph, 'get_split_candidates') as mock_candidates:
            mock_candidates.return_value = [(1, 0.9), (2, 0.85)]
            
            with patch.object(memory, 'split_by_conflict') as mock_split:
                mock_split.return_value = [3, 4]  # New indices after split
                
                conflicts = [{"type": "directional", "nodes": [0, 1]}]
                memory._handle_conflict_split(0, conflicts)
                
                mock_split.assert_called_once_with(0, [(1, 0.9), (2, 0.85)])
    
    def test_error_handling_in_store_episode(self, memory):
        """Test error handling during episode storage."""
        # Mock an error in graph operations
        with patch.object(memory.scalable_graph, 'add_episode_node') as mock_add:
            mock_add.side_effect = Exception("Graph error")
            
            # Should still complete storage despite graph error
            with patch('src.insightspike.core.layers.layer2_memory_manager.logger') as mock_logger:
                result = memory.store_episode("Test content", c_value=0.5)
                
                # Base storage should succeed even if graph fails
                assert len(memory.episodes) > 0
    
    def test_temporal_conflict_handling(self, memory, mock_embedder):
        """Test handling of temporal conflicts."""
        # Add episodes with timestamps
        memory.store_episode("First observation", c_value=0.5, metadata={"timestamp": 1.0})
        
        # Mock temporal conflict detection
        with patch.object(memory.scalable_graph, 'add_episode_node') as mock_add:
            mock_add.return_value = {
                "success": True,
                "conflicts": [{
                    "type": "temporal",
                    "nodes": [0, 1],
                    "similarity": 0.85,
                    "time_diff": 0.5
                }],
                "importance": 0.3
            }
            
            result = memory.store_episode(
                "Different observation",
                c_value=0.5,
                metadata={"timestamp": 1.5}
            )
            
            assert result
            assert len(memory.recent_conflicts) > 0
    
    def test_complex_split_scenario(self, memory, mock_embedder):
        """Test complex splitting scenario with multiple conflicts."""
        # Add base episode with complex content
        complex_text = """
        The system performance increased significantly.
        However, memory usage also increased.
        The overall efficiency decreased due to overhead.
        Further optimization is required.
        """
        
        memory.store_episode(complex_text, c_value=0.8)
        
        # Test splitting into meaningful parts
        parts = memory._analyze_text_for_splits(complex_text)
        assert len(parts) >= 3  # Should split into multiple meaningful parts
        
        # Perform actual split
        new_indices = memory.split_by_conflict(0, [(0, 0.9)])
        
        # Verify split episodes
        for idx in new_indices:
            episode = memory.episodes[idx]
            assert len(episode.text) > 20  # Each part should be meaningful
            assert episode.c < 0.8  # C-value should decay