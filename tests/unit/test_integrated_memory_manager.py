"""
Unit tests for Integrated Memory Manager with proper separation of concerns.
"""

import numpy as np
import pytest

from insightspike.core.episode import Episode
from insightspike.memory.integrated_memory_manager import IntegratedMemoryManager


class TestIntegratedMemoryManager:
    """Test integrated memory manager with separated responsibilities."""
    
    def test_full_pipeline(self):
        """Test the complete query processing pipeline."""
        manager = IntegratedMemoryManager()
        
        # Add some initial episodes
        for i in range(10):
            manager.add_experience(
                text=f"experience {i}",
                vec=np.random.randn(384).astype(np.float32),
                initial_confidence=0.5
            )
        
        # Process a query
        query_vec = np.random.randn(384).astype(np.float32)
        selected, new_insight = manager.process_query(
            query_vec,
            k_search=5,
            generate_insight=True
        )
        
        # Verify results
        assert len(selected) > 0, "Should select some episodes"
        assert new_insight is not None, "Should generate insight"
        assert new_insight.episode_type == "insight"
        assert new_insight.c < 0.5, "New insight should have low confidence"
        
        # Check that manager now has the new insight
        assert len(manager.episodes) == 11  # 10 original + 1 new
    
    def test_separation_of_concerns(self):
        """Test that evaluation and confidence updates are properly separated."""
        manager = IntegratedMemoryManager()
        
        # Add test episodes
        episodes = []
        for i in range(5):
            ep = Episode(
                text=f"test {i}",
                vec=np.random.randn(384).astype(np.float32),
                c=0.5,
                episode_type="experience"
            )
            episodes.append(ep)
            manager.add_episode(ep)
        
        initial_c_values = [ep.c for ep in episodes]
        
        # Use evaluate_and_select which demonstrates clean separation
        selected = manager.evaluate_and_select(episodes)
        
        # Verify that selected episodes have updated confidence
        for ep in selected:
            idx = episodes.index(ep)
            assert ep.c > initial_c_values[idx], "Selected episodes should have higher confidence"
    
    def test_statistics(self):
        """Test statistics gathering."""
        manager = IntegratedMemoryManager()
        
        # Add mixed episodes
        for i in range(5):
            manager.add_experience(
                f"experience {i}",
                np.random.randn(384).astype(np.float32)
            )
        
        for i in range(3):
            insight = Episode(
                text=f"insight {i}",
                vec=np.random.randn(384).astype(np.float32),
                episode_type="insight",
                c=0.3
            )
            manager.add_episode(insight)
        
        stats = manager.get_statistics()
        
        assert stats["total_episodes"] == 8
        assert stats["type_distribution"]["experiences"] == 5
        assert stats["type_distribution"]["insights"] == 3
        assert "confidence_stats" in stats
        assert "pruning_stats" in stats
    
    def test_memory_pruning_integration(self):
        """Test that memory pruning works with the integrated manager."""
        # Create manager with small capacity
        manager = IntegratedMemoryManager(max_episodes=10)
        
        # Add many episodes to trigger pruning
        for i in range(15):
            manager.add_experience(
                f"experience {i}",
                np.random.randn(384).astype(np.float32),
                initial_confidence=0.1 + i * 0.05  # Varying confidence
            )
        
        # Process query to trigger pruning
        query_vec = np.random.randn(384).astype(np.float32)
        manager.process_query(query_vec)
        
        # Should have pruned to stay under limit
        assert len(manager.episodes) <= 10, "Should prune to stay under max_episodes"
    
    def test_reset(self):
        """Test reset functionality."""
        manager = IntegratedMemoryManager()
        
        # Add episodes
        for i in range(5):
            manager.add_experience(
                f"test {i}",
                np.random.randn(384).astype(np.float32)
            )
        
        assert len(manager.episodes) == 5
        
        # Reset
        manager.reset()
        
        assert len(manager.episodes) == 0, "Should clear all episodes"