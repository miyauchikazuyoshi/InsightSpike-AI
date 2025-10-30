"""
Confidence Manager - Centralized C-value Management
===================================================

Manages C-value updates based on episode selection and usage.
Separates evaluation logic from confidence update logic.
"""

import logging
from typing import List, Tuple

from ..core.episode import Episode

logger = logging.getLogger(__name__)


class ConfidenceManager:
    """
    Centralized manager for C-value (confidence) updates.
    
    Responsible for:
    - Updating confidence after episode selection
    - Managing confidence decay for unused episodes
    - Tracking selection history
    """
    
    def __init__(
        self,
        insight_boost: float = 0.2,
        experience_boost: float = 0.1,
        unselected_decay: float = 0.05
    ):
        """
        Initialize confidence manager.
        
        Args:
            insight_boost: Confidence increase for selected insights
            experience_boost: Confidence increase for selected experiences
            unselected_decay: Confidence decrease for unselected insights
        """
        self.insight_boost = insight_boost
        self.experience_boost = experience_boost
        self.unselected_decay = unselected_decay
    
    def update_after_selection(
        self,
        selected: List[Episode],
        candidates: List[Episode]
    ) -> None:
        """
        Update C-values based on selection results.
        
        This is the main entry point for confidence updates after
        any selection process (geDIG, search, etc.).
        
        Args:
            selected: Episodes that were selected/used
            candidates: All candidate episodes considered
        """
        # Update selected episodes
        for episode in selected:
            self._boost_confidence(episode)
        
        # Decay unselected insights
        for episode in candidates:
            # Use 'is' comparison to avoid numpy array ambiguity
            is_selected = any(episode is sel for sel in selected)
            if not is_selected and episode.episode_type == "insight":
                self._decay_confidence(episode)
    
    def _boost_confidence(self, episode: Episode) -> None:
        """
        Increase confidence for a selected episode.
        
        Args:
            episode: Episode that was selected
        """
        if episode.episode_type == "insight":
            # Insight was validated by selection
            episode.increment_confidence(self.insight_boost)
            logger.debug(
                f"Insight selected, C-value increased to {episode.c:.2f}: "
                f"{episode.text[:50]}"
            )
        else:
            # Experience was reconfirmed
            episode.increment_confidence(self.experience_boost)
            logger.debug(
                f"Experience selected, C-value increased to {episode.c:.2f}: "
                f"{episode.text[:50]}"
            )
    
    def _decay_confidence(self, episode: Episode) -> None:
        """
        Decrease confidence for an unselected episode.
        
        Args:
            episode: Episode that was not selected
        """
        episode.decay_confidence(self.unselected_decay)
        logger.debug(
            f"Episode not selected, C-value decreased to {episode.c:.2f}: "
            f"{episode.text[:50]}"
        )
    
    def batch_update(
        self,
        selection_results: List[Tuple[List[Episode], List[Episode]]]
    ) -> None:
        """
        Update confidence for multiple selection results.
        
        Args:
            selection_results: List of (selected, candidates) tuples
        """
        for selected, candidates in selection_results:
            self.update_after_selection(selected, candidates)
    
    def update_from_usage(
        self,
        episode: Episode,
        was_helpful: bool
    ) -> None:
        """
        Update confidence based on explicit usage feedback.
        
        Args:
            episode: Episode that was used
            was_helpful: Whether the episode was helpful
        """
        if was_helpful:
            # Larger boost for explicit positive feedback
            boost = self.insight_boost * 1.5 if episode.episode_type == "insight" else self.experience_boost * 1.5
            episode.increment_confidence(boost)
            logger.info(f"Episode marked helpful, C-value increased to {episode.c:.2f}")
        else:
            # Larger decay for explicit negative feedback
            decay = self.unselected_decay * 2
            episode.decay_confidence(decay)
            logger.info(f"Episode marked unhelpful, C-value decreased to {episode.c:.2f}")
    
    def reset_confidence(
        self,
        episodes: List[Episode],
        default_value: float = 0.5
    ) -> None:
        """
        Reset confidence values to default.
        
        Useful for recovery from problematic states.
        
        Args:
            episodes: Episodes to reset
            default_value: Default confidence value
        """
        for episode in episodes:
            episode.c = default_value
            episode.selection_count = 0
        
        logger.info(f"Reset confidence for {len(episodes)} episodes to {default_value}")
    
    def get_confidence_stats(self, episodes: List[Episode]) -> dict:
        """
        Get confidence statistics for a set of episodes.
        
        Args:
            episodes: Episodes to analyze
            
        Returns:
            Dictionary with confidence statistics
        """
        if not episodes:
            return {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "high_confidence_count": 0,
                "low_confidence_count": 0
            }
        
        c_values = [ep.c for ep in episodes]
        
        return {
            "mean": sum(c_values) / len(c_values),
            "min": min(c_values),
            "max": max(c_values),
            "high_confidence_count": sum(1 for c in c_values if c >= 0.7),
            "low_confidence_count": sum(1 for c in c_values if c < 0.3)
        }