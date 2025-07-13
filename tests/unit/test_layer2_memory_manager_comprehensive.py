"""
Comprehensive tests for Layer 2 Memory Manager
Covers more functionality to improve coverage
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import faiss
import numpy as np
import pytest

from insightspike.core.config import get_config
from insightspike.core.layers.layer2_memory_manager import Episode, L2MemoryManager


class TestL2MemoryManagerCore:
    """Test core L2MemoryManager functionality."""

    @pytest.fixture
    def memory(self):
        """Create a memory manager instance."""
        return L2MemoryManager(dim=8)

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = Mock()
        config.embedding.dimension = 8
        config.memory.nlist = 16
        config.memory.pq_segments = 4
        config.memory.c_value_gamma = 0.9
        config.memory.c_value_min = 0.1
        config.memory.c_value_max = 1.0
        config.memory.index_file = "data/index.faiss"
        config.reasoning.episode_integration_similarity_threshold = 0.85
        config.reasoning.episode_integration_content_threshold = 0.7
        config.reasoning.episode_integration_c_threshold = 0.3
        return config

    def test_init_with_different_configs(self):
        """Test initialization with various configurations."""
        # Test with explicit dimension
        memory1 = L2MemoryManager(dim=16)
        assert memory1.dim == 16

        # Test with config
        with patch(
            "insightspike.core.layers.layer2_memory_manager.get_config"
        ) as mock_get_config:
            mock_config = Mock()
            mock_config.embedding.dimension = 32
            mock_config.memory.nlist = 64
            mock_config.memory.pq_segments = 8
            mock_config.memory.c_value_gamma = 0.8
            mock_config.memory.c_value_min = 0.0
            mock_config.memory.c_value_max = 1.0
            mock_get_config.return_value = mock_config

            memory2 = L2MemoryManager(config=mock_config)
            assert memory2.dim == 32

    def test_episode_class(self):
        """Test Episode class functionality."""
        vec = np.random.rand(8).astype(np.float32)
        text = "Test episode"
        metadata = {"source": "test"}

        episode = Episode(vec, text, c=0.8, metadata=metadata)

        assert np.array_equal(episode.vec, vec)
        assert episode.text == text
        assert episode.c == 0.8
        assert episode.metadata == metadata
        assert "Test episode" in repr(episode)

    def test_store_episode(self, memory):
        """Test storing episodes."""
        with patch(
            "insightspike.core.layers.layer2_memory_manager.get_model"
        ) as mock_model:
            mock_embedder = Mock()
            mock_embedder.encode.return_value = np.random.rand(1, 8).astype(np.float32)
            mock_model.return_value = mock_embedder

            result = memory.store_episode("Test text", c_value=0.6)

            assert result is True
            assert len(memory.episodes) == 1
            assert memory.episodes[0].text == "Test text"
            assert memory.episodes[0].c == 0.6

    def test_search_episodes(self, memory):
        """Test searching episodes."""
        with patch(
            "insightspike.core.layers.layer2_memory_manager.get_model"
        ) as mock_model:
            # Add some episodes first
            vecs = []
            for i in range(5):
                vec = np.random.rand(8).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                vecs.append(vec)
                episode = Episode(vec, f"Episode {i}", c=0.5 + i * 0.1)
                memory.episodes.append(episode)

            # Train index
            memory._train_index()

            # Mock query encoding
            query_vec = vecs[2] + np.random.rand(8) * 0.1  # Similar to episode 2
            query_vec = query_vec / np.linalg.norm(query_vec)
            mock_embedder = Mock()
            mock_embedder.encode.return_value = query_vec.reshape(1, -1)
            mock_model.return_value = mock_embedder

            results = memory.search_episodes("Query", k=3)

            assert len(results) <= 3
            assert all("text" in r for r in results)
            assert all("similarity" in r for r in results)
            assert all("c_value" in r for r in results)
            assert all("weighted_score" in r for r in results)

    def test_update_c_value(self, memory):
        """Test C-value updates."""
        vec = np.random.rand(8).astype(np.float32)
        episode = Episode(vec, "Test", c=0.5)
        memory.episodes.append(episode)

        # Valid update
        result = memory.update_c_value(0, 0.8)
        assert result is True
        assert memory.episodes[0].c == 0.8

        # Invalid index
        result = memory.update_c_value(10, 0.9)
        assert result is False

        # Test clamping
        memory.update_c_value(0, 2.0)  # Above max
        assert memory.episodes[0].c == memory.c_max

        memory.update_c_value(0, -1.0)  # Below min
        assert memory.episodes[0].c == memory.c_min

    def test_get_memory_stats(self, memory):
        """Test memory statistics."""
        # Empty memory
        stats = memory.get_memory_stats()
        assert stats["total_episodes"] == 0
        assert stats["index_trained"] is False

        # Add episodes with different C-values
        for i in range(5):
            vec = np.random.rand(8).astype(np.float32)
            episode = Episode(vec, f"Episode {i}", c=0.3 + i * 0.1)
            memory.episodes.append(episode)

        stats = memory.get_memory_stats()
        assert stats["total_episodes"] == 5
        assert "c_value_mean" in stats
        assert "c_value_std" in stats
        assert "c_value_min" in stats
        assert "c_value_max" in stats
        assert stats["dimension"] == 8

    def test_save_and_load(self, memory):
        """Test saving and loading memory state."""
        # Add some episodes
        for i in range(3):
            vec = np.random.rand(8).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            episode = Episode(vec, f"Episode {i}", c=0.5 + i * 0.1, metadata={"idx": i})
            memory.episodes.append(episode)

        # Train index
        memory._train_index()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_index.faiss"

            # Save
            result = memory.save(save_path)
            assert result is True
            assert save_path.exists()
            assert (save_path.parent / "episodes.json").exists()

            # Create new memory and load
            new_memory = L2MemoryManager(dim=8)
            result = new_memory.load(save_path)
            assert result is True
            assert len(new_memory.episodes) == 3
            assert new_memory.is_trained is True
            assert new_memory.episodes[0].text == "Episode 0"
            assert new_memory.episodes[2].c == 0.7

    def test_build_from_documents(self):
        """Test building memory from documents."""
        with patch(
            "insightspike.core.layers.layer2_memory_manager.get_model"
        ) as mock_model:
            mock_embedder = Mock()
            # Return different embeddings for each document
            embeddings = np.random.rand(3, 8).astype(np.float32)
            for i in range(3):
                embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])
            mock_embedder.encode.return_value = embeddings
            mock_model.return_value = mock_embedder

            docs = ["Doc 1", "Doc 2", "Doc 3"]
            memory = L2MemoryManager.build_from_documents(docs)

            assert len(memory.episodes) == 3
            assert memory.is_trained is True
            assert memory.episodes[0].text == "Doc 1"

    def test_interface_methods(self, memory):
        """Test interface compliance methods."""
        vec = np.random.rand(8).astype(np.float32)

        # Test add_episode
        idx = memory.add_episode(vec, "Test", c_value=0.5)
        assert idx == 0
        assert len(memory.episodes) == 1

        # Test search
        similarities, indices = memory.search(vec, top_k=1)
        assert len(similarities) <= 1
        assert len(indices) <= 1

        # Test update_c_values
        memory.update_c_values([0], [0.1])
        assert memory.episodes[0].c != 0.5  # Should have changed

        # Test update_c
        result = memory.update_c([0], 0.2, eta=0.5)
        assert result is True

    def test_train_index_method(self, memory):
        """Test public train_index method."""
        # Add episodes
        for i in range(10):
            vec = np.random.rand(8).astype(np.float32)
            episode = Episode(vec, f"Episode {i}")
            memory.episodes.append(episode)

        result = memory.train_index()
        assert result is True
        assert memory.is_trained is True

    def test_prune_method(self, memory):
        """Test pruning episodes."""
        # Add episodes with different C-values
        for i in range(10):
            vec = np.random.rand(8).astype(np.float32)
            c_value = 0.1 + i * 0.08  # Range from 0.1 to 0.82
            episode = Episode(vec, f"Episode {i}", c=c_value)
            memory.episodes.append(episode)

        # Prune episodes with C < 0.5
        pruned = memory.prune(c_threshold=0.5)

        assert pruned == 5  # Should remove 5 episodes
        assert len(memory.episodes) == 5
        assert all(ep.c >= 0.5 for ep in memory.episodes)

    def test_merge_method(self, memory):
        """Test merging episodes."""
        # Add episodes
        vecs = []
        for i in range(5):
            vec = np.random.rand(8).astype(np.float32)
            vecs.append(vec)
            episode = Episode(vec, f"Episode {i}", c=0.5 + i * 0.1)
            memory.episodes.append(episode)

        # Merge episodes 1, 2, 3
        merged_idx = memory.merge([1, 2, 3])

        assert merged_idx >= 0
        assert len(memory.episodes) == 3  # 5 - 3 + 1
        merged = memory.episodes[merged_idx]
        assert "Episode 1 | Episode 2 | Episode 3" in merged.text
        assert merged.c == 0.8  # Max of merged episodes

    def test_split_method(self, memory):
        """Test splitting episodes."""
        vec = np.random.rand(8).astype(np.float32)
        text = "First sentence. Second sentence. Third sentence."
        episode = Episode(vec, text, c=0.8)
        memory.episodes.append(episode)

        # Split the episode
        new_indices = memory.split(0)

        assert len(new_indices) == 3  # Three sentences
        assert len(memory.episodes) == 3  # Original removed, 3 added
        assert all("sentence" in memory.episodes[i].text.lower() for i in range(3))

    def test_episode_similarity(self, memory):
        """Test calculating episode similarities."""
        # Add episodes
        for i in range(3):
            vec = np.random.rand(8).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            episode = Episode(vec, f"Episode {i}")
            memory.episodes.append(episode)

        similarities = memory.get_episode_similarity([0, 1, 2])

        assert len(similarities) == 3  # 3 pairs: (0,1), (0,2), (1,2)
        assert all(0 <= s <= 1 for s in similarities)

    def test_episode_complexity(self, memory):
        """Test episode content complexity calculation."""
        vec = np.random.rand(8).astype(np.float32)

        # Simple episode
        simple_text = "Short text."
        episode1 = Episode(vec, simple_text)
        memory.episodes.append(episode1)

        # Complex episode
        complex_text = "This is a much longer text with multiple sentences. It contains various words and concepts. The complexity should be higher."
        episode2 = Episode(vec, complex_text)
        memory.episodes.append(episode2)

        complexity1 = memory.get_episode_content_complexity(0)
        complexity2 = memory.get_episode_content_complexity(1)

        assert 0 <= complexity1 <= 1
        assert 0 <= complexity2 <= 1
        assert complexity2 > complexity1  # Complex text should have higher complexity

    def test_episode_integration(self, memory, mock_config):
        """Test episode integration logic."""
        with patch(
            "insightspike.core.layers.layer2_memory_manager.get_config",
            return_value=mock_config,
        ):
            with patch(
                "insightspike.core.layers.layer2_memory_manager.get_model"
            ) as mock_model:
                # Add initial episode
                vec1 = np.random.rand(8).astype(np.float32)
                vec1 = vec1 / np.linalg.norm(vec1)
                memory.episodes.append(
                    Episode(vec1, "Machine learning is fascinating", c=0.7)
                )

                # Try to add similar episode (should integrate)
                vec2 = vec1 + np.random.rand(8) * 0.01  # Very similar
                vec2 = vec2 / np.linalg.norm(vec2)

                mock_embedder = Mock()
                mock_embedder.encode.return_value = vec2.reshape(1, -1)
                mock_model.return_value = mock_embedder

                result = memory.store_episode(
                    "Machine learning is very fascinating", c_value=0.8
                )

                # Should still have 1 episode (integrated)
                assert len(memory.episodes) == 1
                assert (
                    "Machine learning is fascinating | Machine learning is very fascinating"
                    in memory.episodes[0].text
                )
                assert memory.episodes[0].c == 0.8  # Max of the two

    def test_error_handling(self, memory):
        """Test error handling in various methods."""
        # Test with invalid inputs
        assert memory.update_c_value(-1, 0.5) is False
        assert memory.merge([]) == -1
        assert memory.split(10) == []

        # Test search with empty memory
        results = memory.search_episodes("test")
        assert results == []

        # Test save/load with invalid paths
        assert memory.save(Path("/invalid/path/test.faiss")) is False
        assert memory.load(Path("/nonexistent/test.faiss")) is False

    def test_linear_search_fallback(self, memory):
        """Test linear search for small datasets."""
        with patch(
            "insightspike.core.layers.layer2_memory_manager.get_model"
        ) as mock_model:
            # Add just one episode (too small for index training)
            vec = np.random.rand(8).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            episode = Episode(vec, "Single episode", c=0.7)
            memory.episodes.append(episode)

            # Mock query encoding
            query_vec = vec + np.random.rand(8) * 0.1
            query_vec = query_vec / np.linalg.norm(query_vec)
            mock_embedder = Mock()
            mock_embedder.encode.return_value = query_vec.reshape(1, -1)
            mock_model.return_value = mock_embedder

            results = memory.search_episodes("Query")

            assert len(results) == 1
            assert results[0]["text"] == "Single episode"
            assert memory.is_trained is False  # Should not train with 1 episode
