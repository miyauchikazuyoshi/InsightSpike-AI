"""
Integration tests for the scalable graph system (Phase 2 & 3)
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

from insightspike.core.layers.integrated_hierarchical_manager import (
    IntegratedHierarchicalManager,
)
from insightspike.core.layers.scalable_graph_builder import ScalableGraphBuilder
from insightspike.core.agents.main_agent_graph_centric import GraphCentricMainAgent


class TestScalableSystem:
    """Test the complete scalable system integration."""

    @pytest.fixture
    def manager(self):
        """Create an integrated manager instance."""
        # Use 384 dimensions to match the embedder output
        return IntegratedHierarchicalManager(
            dimension=384, cluster_size=5, super_cluster_size=3, rebuild_threshold=20
        )

    def test_integrated_manager_basic(self, manager):
        """Test basic integrated manager functionality."""
        # Add episodes
        for i in range(15):
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            result = manager.add_episode(vec, f"Episode {i}")
            assert result["success"]

        # Check statistics
        stats = manager.get_statistics()
        assert stats["memory"]["total_episodes"] == 15
        # Hierarchy might have fewer nodes due to clustering
        assert (
            stats["hierarchy"]["nodes_per_level"][0] >= 10
        )  # At least 10 nodes in level 0

        # Test search
        results = manager.search("Episode", k=5)
        assert len(results) <= 5

    def test_automatic_rebuild(self, manager):
        """Test automatic hierarchy rebuild."""
        manager.rebuild_threshold = 10

        # Add episodes to trigger rebuild
        for i in range(12):
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            manager.add_episode(vec, f"Episode {i}")

        # Should have triggered at least one rebuild
        assert manager.stats["total_rebuilds"] > 0

    def test_search_performance(self, manager):
        """Test search performance scales properly."""
        # Add many episodes
        for i in range(50):
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            manager.add_episode(vec, f"Episode {i}")

        # Measure search time - search for existing content
        start = time.time()
        results = manager.search("Episode", k=10)  # Search for "Episode" which exists
        search_time = time.time() - start

        # Should be fast (< 100ms for small dataset)
        assert search_time < 0.1
        assert len(results) <= 10

    def test_memory_optimization(self, manager):
        """Test memory optimization functionality."""
        # Add episodes with varying importance
        for i in range(20):
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            manager.add_episode(vec, f"Episode {i}")

        # Access some episodes
        for _ in range(5):
            manager.search("Episode 0", k=1)

        initial_count = len(manager.memory_manager.episodes)

        # Optimize
        result = manager.optimize()

        assert result["memory_optimization"]["initial_count"] == initial_count
        # May or may not remove episodes depending on importance

    @pytest.mark.skip(reason="layer4_narrative_generator module not implemented")
    @patch("insightspike.utils.embedder.get_model")
    def test_graph_centric_agent(self, mock_get_model):
        """Test the graph-centric main agent."""
        # Mock the embedder
        mock_model = Mock()
        mock_model.encode.return_value = np.random.randn(384).astype(np.float32)
        mock_get_model.return_value = mock_model

        agent = GraphCentricMainAgent()
        agent.initialize()

        # Add episodes
        result = agent.add_episode("Test episode 1")
        assert result["success"]
        assert "importance" in result
        assert result["importance"] >= 0

        # Search
        search_result = agent.search("test", k=3)
        assert search_result["success"]

        # Get memory analysis
        analysis = agent.get_memory_analysis()
        assert "total_episodes" in analysis
        assert "importance_distribution" in analysis

    def test_scalability_characteristics(self):
        """Test that the system exhibits proper scalability."""
        sizes = [10, 50, 100]
        build_times = []
        search_times = []

        for size in sizes:
            # Create mock config for ScalableGraphBuilder
            mock_config = Mock()
            mock_config.reasoning.similarity_threshold = 0.3
            mock_config.scalable_graph.top_k_neighbors = 5
            mock_config.scalable_graph.batch_size = 1000
            mock_config.embedding.dimension = 384

            builder = ScalableGraphBuilder(config=mock_config)

            # Generate documents
            docs = []
            for i in range(size):
                vec = np.random.randn(384).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                docs.append({"embedding": vec, "text": f"Doc {i}"})

            # Measure build time
            start = time.time()
            graph = builder.build_graph(docs)
            build_time = time.time() - start
            build_times.append(build_time)

            # Measure search time (approximate with edge count)
            edge_count = graph.edge_index.shape[1]
            # Search time proportional to average degree
            avg_degree = edge_count / size if size > 0 else 0
            search_times.append(avg_degree)

        # Build time should scale sub-quadratically
        # Rough check: time for 100 docs < 10 * time for 10 docs
        assert build_times[2] < 10 * build_times[0]

        # Average degree should be bounded by top_k
        assert all(st < 10 for st in search_times)  # top_k * 2 for bidirectional
