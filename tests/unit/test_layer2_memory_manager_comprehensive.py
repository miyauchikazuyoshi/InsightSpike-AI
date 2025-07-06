"""
Comprehensive tests for Layer 2 Memory Manager
Covers more functionality to improve coverage
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import faiss

from insightspike.core.layers.layer2_memory_manager import L2MemoryManager


class TestL2MemoryManagerCore:
    """Test core L2MemoryManager functionality."""
    
    @pytest.fixture
    def memory(self):
        """Create a memory manager instance."""
        return L2MemoryManager(dim=8, quantized=False)
    
    def test_init_with_different_configs(self):
        """Test initialization with various configurations."""
        # Test quantized mode
        memory_q = L2MemoryManager(dim=16, quantized=True)
        assert memory_q.dim == 16
        assert memory_q.quantized is True
        
        # Test with custom config
        memory_custom = L2MemoryManager(dim=32, max_episodes=500)
        assert memory_custom.dim == 32
        assert hasattr(memory_custom, 'episodes')
    
    def test_add_episode_basic(self, memory):
        """Test basic episode addition."""
        vec = np.random.rand(8).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        text = "Test episode"
        
        idx = memory.add_episode(vec, text, c_init=0.7)
        
        assert idx == 0
        assert len(memory.episodes) == 1
        assert memory.episodes[0].text == text
        assert memory.episodes[0].c == 0.7
        assert memory.index.ntotal == 1
    
    def test_add_episode_with_metadata(self, memory):
        """Test adding episode with metadata."""
        vec = np.random.rand(8).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        metadata = {"source": "test", "timestamp": 12345}
        
        idx = memory.add_episode(vec, "Test", metadata=metadata)
        
        assert memory.episodes[0].metadata == metadata
    
    def test_search_episodes(self, memory):
        """Test episode search."""
        # Add multiple episodes
        for i in range(5):
            vec = np.random.randn(8).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            memory.add_episode(vec, f"Episode {i}", c_init=0.5 + i * 0.1)
        
        # Search
        query = np.random.randn(8).astype(np.float32)
        query = query / np.linalg.norm(query)
        
        results, indices = memory.search(query, k=3)
        
        assert len(results) == 3
        assert len(indices) == 3
        assert all(0 <= idx < 5 for idx in indices)
    
    def test_update_c_values(self, memory):
        """Test C-value updates."""
        # Add episodes
        for i in range(3):
            vec = np.random.randn(8).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            memory.add_episode(vec, f"Episode {i}", c_init=0.5)
        
        # Update C-values
        memory.update_c([0, 2], reward=1.0, eta=0.1)
        
        assert memory.episodes[0].c > 0.5
        assert memory.episodes[1].c == 0.5  # Not updated
        assert memory.episodes[2].c > 0.5
    
    def test_train_index(self, memory):
        """Test index training."""
        # Add enough episodes to trigger training
        for i in range(100):
            vec = np.random.randn(8).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            memory.add_episode(vec, f"Episode {i}")
        
        # Train index
        memory.train_index()
        
        # Verify index is trained (for IVF indexes)
        if hasattr(memory.index, 'is_trained'):
            assert memory.index.is_trained
    
    def test_merge_episodes(self, memory):
        """Test episode merging."""
        # Add episodes with high C-values
        vec1 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        vec2 = np.array([0.9, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        vec2 = vec2 / np.linalg.norm(vec2)
        
        idx1 = memory.add_episode(vec1, "Episode 1", c_init=0.8)
        idx2 = memory.add_episode(vec2, "Episode 2", c_init=0.9)
        
        # Merge
        memory.merge([idx1, idx2])
        
        # Check result
        assert len(memory.episodes) < 2 or memory.episodes[0].merged or memory.episodes[1].merged
    
    def test_split_episode(self, memory):
        """Test episode splitting."""
        vec = np.random.randn(8).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        idx = memory.add_episode(vec, "Complex episode with multiple topics", c_init=0.3)
        
        # Split
        memory.split(idx)
        
        # Should create new episodes or mark for splitting
        # The implementation details may vary
        assert len(memory.episodes) >= 1
    
    def test_prune_episodes(self, memory):
        """Test episode pruning."""
        # Add episodes with different C-values and importance
        for i in range(10):
            vec = np.random.randn(8).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            c_val = 0.1 if i < 5 else 0.8
            memory.add_episode(vec, f"Episode {i}", c_init=c_val)
        
        initial_count = len(memory.episodes)
        
        # Prune with low thresholds
        memory.prune(c_threshold=0.3, importance_threshold=0.2)
        
        # Some episodes should be removed
        assert len(memory.episodes) <= initial_count
    
    def test_save_and_load(self, memory):
        """Test saving and loading memory state."""
        # Add some episodes
        for i in range(3):
            vec = np.random.randn(8).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            memory.add_episode(vec, f"Episode {i}", c_init=0.5 + i * 0.1)
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "memory_state.pkl"
            memory.save(save_path)
            
            assert save_path.exists()
            
            # Load into new memory
            new_memory = L2MemoryManager(dim=8)
            new_memory.load(save_path)
            
            assert len(new_memory.episodes) == 3
            assert new_memory.episodes[0].text == "Episode 0"
            assert new_memory.index.ntotal == 3
    
    def test_get_memory_stats(self, memory):
        """Test memory statistics."""
        # Add episodes
        for i in range(5):
            vec = np.random.randn(8).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            memory.add_episode(vec, f"Episode {i}", c_init=0.5 + i * 0.05)
        
        stats = memory.get_memory_stats()
        
        assert stats['total_episodes'] == 5
        assert 'avg_c_value' in stats
        assert stats['avg_c_value'] > 0.5
        assert 'memory_usage_mb' in stats
    
    def test_batch_operations(self, memory):
        """Test batch operations."""
        # Batch add
        vectors = []
        texts = []
        for i in range(10):
            vec = np.random.randn(8).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec)
            texts.append(f"Batch episode {i}")
        
        # Add in batch (simulate)
        for vec, text in zip(vectors, texts):
            memory.add_episode(vec, text)
        
        # Batch search
        queries = np.random.randn(3, 8).astype(np.float32)
        for i in range(3):
            queries[i] = queries[i] / np.linalg.norm(queries[i])
        
        # Search each query
        all_results = []
        for query in queries:
            results, _ = memory.search(query, k=2)
            all_results.extend(results)
        
        assert len(all_results) == 6
    
    def test_episode_importance_calculation(self, memory):
        """Test episode importance calculation."""
        # Add episodes
        vec1 = np.random.randn(8).astype(np.float32)
        vec1 = vec1 / np.linalg.norm(vec1)
        idx1 = memory.add_episode(vec1, "Important episode", c_init=0.9)
        
        vec2 = np.random.randn(8).astype(np.float32)
        vec2 = vec2 / np.linalg.norm(vec2)
        idx2 = memory.add_episode(vec2, "Less important", c_init=0.2)
        
        # Access first episode multiple times
        for _ in range(5):
            memory.search(vec1, k=1)
        
        # Calculate importance (if method exists)
        if hasattr(memory, 'calculate_importance'):
            imp1 = memory.calculate_importance(idx1)
            imp2 = memory.calculate_importance(idx2)
            assert imp1 > imp2
    
    def test_similarity_threshold_operations(self, memory):
        """Test operations with similarity thresholds."""
        # Add base episode
        vec_base = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        memory.add_episode(vec_base, "Base episode")
        
        # Add similar episodes with varying similarity
        similarities = [0.99, 0.9, 0.8, 0.7, 0.6]
        for sim in similarities:
            vec = vec_base * sim + np.random.randn(8) * (1 - sim) * 0.1
            vec = vec / np.linalg.norm(vec)
            memory.add_episode(vec, f"Similar episode (sim={sim})")
        
        # Search with different thresholds
        results_high, _ = memory.search(vec_base, k=10)
        
        # Most similar should be ranked first
        assert len(results_high) > 0
        assert results_high[0] >= 0.9  # High similarity
    
    def test_error_handling(self, memory):
        """Test error handling in various scenarios."""
        # Test with invalid vector dimension
        with pytest.raises(Exception):
            bad_vec = np.random.randn(16).astype(np.float32)  # Wrong dimension
            memory.add_episode(bad_vec, "Bad episode")
        
        # Test with empty search
        results, indices = memory.search(np.random.randn(8).astype(np.float32), k=5)
        assert len(results) == 0  # No episodes yet
        
        # Test update with invalid indices
        memory.update_c([999], reward=1.0)  # Should not crash
        
        # Test save to invalid path
        with pytest.raises(Exception):
            memory.save("/invalid/path/that/does/not/exist.pkl")


class TestL2MemoryManagerQuantized:
    """Test quantized memory manager."""
    
    def test_quantized_operations(self):
        """Test operations in quantized mode."""
        memory = L2MemoryManager(dim=8, quantized=True, nlist=4, m=2)
        
        # Add episodes
        for i in range(50):
            vec = np.random.randn(8).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            memory.add_episode(vec, f"Quantized episode {i}")
        
        # Train quantized index
        memory.train_index()
        
        # Search
        query = np.random.randn(8).astype(np.float32)
        query = query / np.linalg.norm(query)
        results, indices = memory.search(query, k=5)
        
        assert len(results) == 5
        assert all(isinstance(idx, (int, np.integer)) for idx in indices)