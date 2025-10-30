"""
Insight Generator with C-value Propagation
==========================================

Generates insights from episode combinations.
New insights inherit confidence based on source episode confidence.
"""

import logging
from typing import List, Optional

import numpy as np

from ..core.episode import Episode

logger = logging.getLogger(__name__)


class InsightGenerator:
    """
    Generates insights from episode combinations.
    
    Key principles:
    - No weights or C-values used in vector integration (pure synthesis)
    - Generated insight's C-value depends on source episodes
    - Low confidence materials produce even lower confidence insights
    """
    
    def __init__(self):
        """Initialize insight generator."""
        pass
    
    def generate(
        self,
        selected_episodes: List[Episode],
        insight_text: Optional[str] = None
    ) -> Episode:
        """
        Generate insight from selected episodes.
        
        Pure vector integration without weights or C-values.
        Resulting C-value reflects source confidence.
        
        Args:
            selected_episodes: Episodes to combine
            insight_text: Optional custom text for insight
            
        Returns:
            New insight episode
        """
        if not selected_episodes:
            raise ValueError("Cannot generate insight from empty episode list")
        
        # Pure vector integration (simple average)
        vectors = [ep.vec for ep in selected_episodes]
        insight_vec = np.mean(vectors, axis=0).astype(np.float32)
        
        # Generate or use provided text
        if insight_text is None:
            insight_text = self._synthesize_text(selected_episodes)
        
        # Calculate initial confidence based on sources
        initial_confidence = self._calculate_insight_confidence(selected_episodes)
        
        # Create new insight episode
        insight = Episode(
            text=insight_text,
            vec=insight_vec,
            c=initial_confidence,
            episode_type="insight",
            metadata={
                "source_episodes": [ep.text[:100] for ep in selected_episodes],
                "source_c_values": [ep.c for ep in selected_episodes],
                "generation_method": "mean_integration"
            }
        )
        
        logger.info(
            f"Generated insight with C={initial_confidence:.2f} from "
            f"{len(selected_episodes)} episodes"
        )
        
        return insight
    
    def _calculate_insight_confidence(self, source_episodes: List[Episode]) -> float:
        """
        Calculate insight confidence from source episodes.
        
        Low confidence sources produce even lower confidence insights,
        modeling the human doubt/worry process.
        
        Args:
            source_episodes: Source episodes
            
        Returns:
            Initial confidence value for the insight
        """
        if not source_episodes:
            return 0.3  # Default low confidence
        
        # Calculate average confidence of sources
        source_c_values = [ep.c for ep in source_episodes]
        avg_c = np.mean(source_c_values)
        
        # Propagation rules (modeling doubt cascade)
        if avg_c > 0.7:
            # High confidence sources → moderate insight
            return 0.4
        elif avg_c > 0.5:
            # Medium confidence sources → lower insight
            return 0.3
        elif avg_c > 0.3:
            # Low confidence sources → very low insight
            return 0.2
        else:
            # Very low confidence sources → minimal insight
            # This is the "worry state" - insights from doubts
            return 0.1
    
    def _synthesize_text(self, episodes: List[Episode]) -> str:
        """
        Synthesize text description from episodes.
        
        Args:
            episodes: Source episodes
            
        Returns:
            Synthesized text
        """
        if not episodes:
            return "Empty insight"
        
        # Simple concatenation approach
        texts = []
        for i, ep in enumerate(episodes[:3]):  # Limit to first 3
            preview = ep.text[:100]
            if len(ep.text) > 100:
                preview += "..."
            texts.append(f"[{i+1}] {preview}")
        
        synthesis = "Insight from: " + " | ".join(texts)
        
        # Add confidence indicator
        avg_c = np.mean([ep.c for ep in episodes])
        if avg_c < 0.3:
            synthesis = "[Low confidence] " + synthesis
        
        return synthesis
    
    def generate_batch(
        self,
        episode_combinations: List[List[Episode]]
    ) -> List[Episode]:
        """
        Generate multiple insights from combinations.
        
        Args:
            episode_combinations: List of episode combinations
            
        Returns:
            List of generated insights
        """
        insights = []
        for combination in episode_combinations:
            try:
                insight = self.generate(combination)
                insights.append(insight)
            except Exception as e:
                logger.error(f"Failed to generate insight: {e}")
        
        return insights
    
    def weighted_integration(
        self,
        episodes: List[Episode],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Alternative integration method with custom weights.
        
        Note: This is for experimental purposes. Default is simple mean.
        
        Args:
            episodes: Episodes to integrate
            weights: Optional weights for each episode
            
        Returns:
            Integrated vector
        """
        vectors = np.array([ep.vec for ep in episodes])
        
        if weights is not None:
            if len(weights) != len(episodes):
                raise ValueError("Weights must match number of episodes")
            # Normalize weights
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            # Weighted average
            integrated = np.sum(vectors * weights[:, np.newaxis], axis=0)
        else:
            # Simple average
            integrated = np.mean(vectors, axis=0)
        
        return integrated.astype(np.float32)