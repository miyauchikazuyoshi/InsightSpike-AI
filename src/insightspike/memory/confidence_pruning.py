"""
Confidence-based Memory Pruning
================================

Manages memory by pruning low-confidence episodes.
Adaptive thresholds based on memory pressure.
"""

import logging
import time
from typing import List, Optional

import numpy as np

from ..core.episode import Episode

logger = logging.getLogger(__name__)


class ConfidenceBasedPruning:
    """
    Memory management based on C-values (confidence scores).
    
    Strategies:
    - Remove old, low-confidence insights
    - Adaptive threshold based on memory pressure
    - Preserve high-value experiences
    """
    
    def __init__(
        self,
        min_confidence: float = 0.2,
        max_episodes: int = 10000,
        insight_ttl_hours: float = 1.0
    ):
        """
        Initialize confidence-based pruning.
        
        Args:
            min_confidence: Minimum C-value to keep (default 0.2)
            max_episodes: Maximum number of episodes (default 10000)
            insight_ttl_hours: Time-to-live for insights in hours (default 1.0)
        """
        self.min_confidence = min_confidence
        self.max_episodes = max_episodes
        self.insight_ttl_seconds = insight_ttl_hours * 3600
    
    def prune_low_confidence(
        self,
        episodes: List[Episode],
        force: bool = False
    ) -> List[Episode]:
        """
        Remove low-confidence episodes.
        
        Insights that are old and unused get pruned.
        Experiences are generally preserved.
        
        Args:
            episodes: List of episodes
            force: Force pruning even if under memory limit
            
        Returns:
            Pruned episode list
        """
        if not force and len(episodes) < self.max_episodes * 0.8:
            # No pruning needed if under 80% capacity
            return episodes
        
        current_time = time.time()
        pruned = []
        removed_insights = 0
        removed_experiences = 0
        
        for episode in episodes:
            # Check if episode should be pruned
            if self._should_prune(episode, current_time):
                if episode.episode_type == "insight":
                    removed_insights += 1
                else:
                    removed_experiences += 1
                logger.debug(
                    f"Pruning {episode.episode_type} with C={episode.c:.2f}: "
                    f"{episode.text[:50]}"
                )
            else:
                pruned.append(episode)
        
        if removed_insights > 0 or removed_experiences > 0:
            logger.info(
                f"Pruned {removed_insights} insights and "
                f"{removed_experiences} experiences"
            )
        
        return pruned
    
    def _should_prune(self, episode: Episode, current_time: float) -> bool:
        """
        Determine if an episode should be pruned.
        
        Args:
            episode: Episode to check
            current_time: Current timestamp
            
        Returns:
            True if should be pruned
        """
        # Never prune high-confidence episodes
        if episode.c >= 0.7:
            return False
        
        # Check insights for age and confidence
        if episode.episode_type == "insight":
            age = current_time - episode.creation_time
            
            # Old and low confidence
            if age > self.insight_ttl_seconds and episode.c < self.min_confidence:
                return True
            
            # Very low confidence and never selected
            if episode.c < 0.15 and episode.selection_count == 0:
                return True
        
        # Experiences: only prune if very low confidence and old
        elif episode.episode_type == "experience":
            age = current_time - episode.creation_time
            if age > self.insight_ttl_seconds * 10 and episode.c < 0.1:
                return True
        
        return False
    
    def adaptive_threshold(
        self,
        episodes: List[Episode],
        target_size: Optional[int] = None
    ) -> List[Episode]:
        """
        Dynamically adjust threshold based on memory pressure.
        
        Args:
            episodes: List of episodes
            target_size: Target number of episodes (defaults to max_episodes)
            
        Returns:
            Pruned episode list
        """
        target = target_size or self.max_episodes
        
        if len(episodes) <= target:
            return episodes
        
        # Need to remove episodes
        num_to_remove = len(episodes) - target
        
        # Sort by pruning priority
        sorted_episodes = sorted(
            episodes,
            key=lambda ep: self._pruning_priority(ep)
        )
        
        # Keep only the top episodes
        kept = sorted_episodes[num_to_remove:]
        
        logger.info(
            f"Adaptive pruning removed {num_to_remove} episodes "
            f"to reach target of {target}"
        )
        
        return kept
    
    def _pruning_priority(self, episode: Episode) -> float:
        """
        Calculate pruning priority (higher = keep).
        
        Args:
            episode: Episode to score
            
        Returns:
            Priority score
        """
        # Base score is confidence
        score = episode.c
        
        # Bonus for experiences (they're real data)
        if episode.episode_type == "experience":
            score += 0.3
        
        # Bonus for being selected
        score += min(0.3, episode.selection_count * 0.05)
        
        # Penalty for age (insights only)
        if episode.episode_type == "insight":
            age_hours = (time.time() - episode.creation_time) / 3600
            age_penalty = min(0.5, age_hours * 0.1)
            score -= age_penalty
        
        return score
    
    def get_statistics(self, episodes: List[Episode]) -> dict:
        """
        Get pruning statistics.
        
        Args:
            episodes: List of episodes
            
        Returns:
            Statistics dictionary
        """
        if not episodes:
            return {
                "total": 0,
                "insights": 0,
                "experiences": 0,
                "avg_confidence": 0.0,
                "low_confidence_count": 0,
                "prunable_count": 0
            }
        
        current_time = time.time()
        
        insights = [ep for ep in episodes if ep.episode_type == "insight"]
        experiences = [ep for ep in episodes if ep.episode_type == "experience"]
        low_conf = [ep for ep in episodes if ep.c < self.min_confidence]
        prunable = [ep for ep in episodes if self._should_prune(ep, current_time)]
        
        return {
            "total": len(episodes),
            "insights": len(insights),
            "experiences": len(experiences),
            "avg_confidence": np.mean([ep.c for ep in episodes]),
            "low_confidence_count": len(low_conf),
            "prunable_count": len(prunable),
            "memory_usage_ratio": len(episodes) / self.max_episodes
        }
    
    def recommend_action(self, episodes: List[Episode]) -> str:
        """
        Recommend pruning action based on current state.
        
        Args:
            episodes: List of episodes
            
        Returns:
            Recommended action string
        """
        stats = self.get_statistics(episodes)
        
        if stats["memory_usage_ratio"] > 0.9:
            return "urgent_prune"  # Over 90% capacity
        elif stats["memory_usage_ratio"] > 0.7:
            return "consider_prune"  # Over 70% capacity
        elif stats["prunable_count"] > stats["total"] * 0.3:
            return "many_prunable"  # Many low-value episodes
        else:
            return "no_action"  # System healthy