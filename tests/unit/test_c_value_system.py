"""
Unit tests for C-value (confidence) system.
"""

import time
import numpy as np
import pytest

from insightspike.core.episode import Episode
from insightspike.search.similarity_search import SimilaritySearch
from insightspike.evaluation.gedig_evaluator import GeDIGEvaluator
from insightspike.generation.insight_generator import InsightGenerator
from insightspike.confidence.confidence_manager import ConfidenceManager
from insightspike.memory.confidence_pruning import ConfidenceBasedPruning
from insightspike.memory.experience_tracker import ExperienceTracker
from insightspike.memory.integrated_memory_manager import IntegratedMemoryManager


class TestEpisodeConfidence:
    """Test Episode class confidence features."""
    
    def test_episode_type_initial_confidence(self):
        """Test that episodes get correct initial confidence based on type."""
        # Experience episode
        exp = Episode(
            text="test experience",
            vec=np.random.randn(384).astype(np.float32),
            episode_type="experience"
        )
        assert exp.c == 0.5  # Experiences start at 0.5
        
        # Insight episode
        insight = Episode(
            text="test insight",
            vec=np.random.randn(384).astype(np.float32),
            episode_type="insight"
        )
        assert insight.c == 0.3  # Insights start at 0.3
        
        # Contradiction episode
        contradiction = Episode(
            text="test contradiction",
            vec=np.random.randn(384).astype(np.float32),
            episode_type="contradiction"
        )
        assert contradiction.c == 0.4  # Contradictions at 0.4
    
    def test_confidence_increment_decrement(self):
        """Test confidence value changes."""
        ep = Episode(
            text="test",
            vec=np.random.randn(384).astype(np.float32),
            c=0.5
        )
        
        # Test increment
        ep.increment_confidence(0.2)
        assert ep.c == 0.7
        assert ep.selection_count == 1
        
        # Test max cap
        ep.increment_confidence(0.5)
        assert ep.c == 1.0  # Capped at 1.0
        
        # Test decrement
        ep.decay_confidence(0.3)
        assert ep.c == 0.7
        
        # Test min cap
        ep.decay_confidence(1.0)
        assert ep.c == 0.1  # Minimum is 0.1


class TestSimilaritySearch:
    """Test similarity search without C-value influence."""
    
    def test_search_ignores_c_values(self):
        """Test that search uses pure similarity, not C-values."""
        # Create episodes with different C-values
        query_vec = np.array([1.0, 0.0, 0.0]).astype(np.float32)
        
        episodes = [
            Episode(
                text="high confidence but dissimilar",
                vec=np.array([0.0, 1.0, 0.0]).astype(np.float32),
                c=0.9
            ),
            Episode(
                text="low confidence but similar",
                vec=np.array([0.9, 0.1, 0.0]).astype(np.float32),
                c=0.1
            ),
        ]
        
        searcher = SimilaritySearch()
        results = searcher.search(query_vec, episodes, k=2)
        
        # Should return similar vector first, despite low C-value
        assert results[0].text == "low confidence but similar"
        assert results[1].text == "high confidence but dissimilar"


class TestGeDIGEvaluator:
    """Test geDIG evaluation and C-value updates."""
    
    def test_gedig_pure_evaluation(self):
        """Test that geDIG doesn't use C-values in calculation."""
        evaluator = GeDIGEvaluator(k=0.5)
        
        # Create episodes with different C-values
        episodes = [
            Episode(
                text="episode 1",
                vec=np.array([1.0, 0.0, 0.0]).astype(np.float32),
                c=0.9
            ),
            Episode(
                text="episode 2",
                vec=np.array([0.0, 1.0, 0.0]).astype(np.float32),
                c=0.1
            ),
        ]
        
        # Calculate geDIG
        score1 = evaluator.evaluate(episodes)
        
        # Change C-values
        episodes[0].c = 0.1
        episodes[1].c = 0.9
        
        # Calculate again
        score2 = evaluator.evaluate(episodes)
        
        # Scores should be identical (C-values not used)
        assert score1 == score2
    
    def test_confidence_update_separation(self):
        """Test that evaluation and confidence updates are separated."""
        evaluator = GeDIGEvaluator()
        confidence_manager = ConfidenceManager()
        
        # Create candidates
        candidates = [
            Episode(
                text=f"episode {i}",
                vec=np.random.randn(384).astype(np.float32),
                c=0.5,
                episode_type="insight" if i % 2 == 0 else "experience"
            )
            for i in range(5)
        ]
        
        initial_c_values = [ep.c for ep in candidates]
        
        # Step 1: Pure evaluation (no side effects)
        selected, all_candidates = evaluator.rerank(candidates, top_k=3)
        
        # C-values should not change after evaluation
        for i, ep in enumerate(candidates):
            assert ep.c == initial_c_values[i], "Evaluation should not change C-values"
        
        # Step 2: Separate confidence update
        confidence_manager.update_after_selection(selected, all_candidates)
        
        # Now C-values should be updated
        for ep in selected:
            idx = candidates.index(ep)
            if ep.episode_type == "insight":
                assert ep.c > initial_c_values[idx]
            else:
                assert ep.c >= initial_c_values[idx]


class TestInsightGenerator:
    """Test insight generation with C-value propagation."""
    
    def test_confidence_propagation(self):
        """Test that low confidence sources create lower confidence insights."""
        generator = InsightGenerator()
        
        # High confidence sources
        high_conf_episodes = [
            Episode(
                text=f"high conf {i}",
                vec=np.random.randn(384).astype(np.float32),
                c=0.8
            )
            for i in range(3)
        ]
        
        high_insight = generator.generate(high_conf_episodes)
        assert high_insight.c == 0.4  # High sources → 0.4
        
        # Low confidence sources
        low_conf_episodes = [
            Episode(
                text=f"low conf {i}",
                vec=np.random.randn(384).astype(np.float32),
                c=0.25
            )
            for i in range(3)
        ]
        
        low_insight = generator.generate(low_conf_episodes)
        assert low_insight.c == 0.1  # Very low sources → 0.1 (worry state)


# Worry state tests removed as feature was dropped


class TestConfidencePruning:
    """Test memory management based on confidence."""
    
    def test_prune_old_low_confidence(self):
        """Test pruning of old, low-confidence episodes."""
        pruner = ConfidenceBasedPruning(
            min_confidence=0.2,
            insight_ttl_hours=0.0001  # Very short TTL for testing
        )
        
        # Create old low-confidence insight
        old_insight = Episode(
            text="old insight",
            vec=np.random.randn(384).astype(np.float32),
            c=0.15,
            episode_type="insight"
        )
        old_insight.creation_time = time.time() - 1000  # Old
        
        # Create new low-confidence insight
        new_insight = Episode(
            text="new insight",
            vec=np.random.randn(384).astype(np.float32),
            c=0.15,
            episode_type="insight"
        )
        
        # Create high-confidence episode
        good_episode = Episode(
            text="good episode",
            vec=np.random.randn(384).astype(np.float32),
            c=0.8,
            episode_type="experience"
        )
        
        episodes = [old_insight, new_insight, good_episode]
        pruned = pruner.prune_low_confidence(episodes, force=True)
        
        # Should prune old insight but keep others
        assert len(pruned) == 2
        assert old_insight not in pruned
        assert new_insight in pruned
        assert good_episode in pruned


class TestExperienceTracker:
    """Test experience tracking and confidence growth."""
    
    def test_repeated_experience_confidence(self):
        """Test that repeated experiences increase confidence."""
        tracker = ExperienceTracker()
        
        # First experience
        ep1 = tracker.add_experience(
            state="position_0_0",
            action="move_right",
            result="success"
        )
        assert ep1.c == 0.5  # First time
        
        # Second same experience
        ep2 = tracker.add_experience(
            state="position_0_0",
            action="move_right",
            result="success"
        )
        assert ep2.c == 0.7  # Second time
        
        # Third same experience
        ep3 = tracker.add_experience(
            state="position_0_0",
            action="move_right",
            result="success"
        )
        assert ep3.c == 0.85  # Third time
    
    def test_experience_statistics(self):
        """Test experience tracking statistics."""
        tracker = ExperienceTracker()
        
        # Add various experiences
        for i in range(3):
            tracker.add_experience(f"state_{i}", "action_a", "success")
        
        # Add repeated experience
        for i in range(5):
            tracker.add_experience("state_0", "action_b", "success")
        
        stats = tracker.get_statistics()
        
        assert stats["unique_experiences"] == 4  # 3 + 1 unique
        assert stats["total_repetitions"] == 8  # 3 + 5 total
        assert stats["max_repetitions"] == 5