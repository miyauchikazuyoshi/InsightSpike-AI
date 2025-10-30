"""
Experience Tracker - Confidence Growth Through Repetition
=========================================================

Tracks repeated experiences and increases confidence accordingly.
Models how repeated experiences build certainty.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..core.episode import Episode

logger = logging.getLogger(__name__)


class ExperienceTracker:
    """
    Tracks experiences and manages confidence growth.
    
    Key principle: Repeated experiences increase confidence.
    - First experience: C=0.5
    - Second experience: C=0.7
    - Third+ experience: C approaches 1.0
    """
    
    def __init__(self):
        """Initialize experience tracker."""
        self.experience_counts: Dict[str, int] = {}
        self.experience_episodes: Dict[str, Episode] = {}
    
    def add_experience(
        self,
        state: Any,
        action: Any,
        result: Any,
        vector: Optional[np.ndarray] = None,
        custom_text: Optional[str] = None
    ) -> Episode:
        """
        Add an experience and manage confidence.
        
        Repeated experiences get higher confidence.
        
        Args:
            state: Current state
            action: Action taken
            result: Result of action
            vector: Optional vector representation
            custom_text: Optional custom text description
            
        Returns:
            Episode with appropriate confidence
        """
        # Create experience key
        key = self._create_key(state, action, result)
        
        # Check if this is a repeated experience
        if key in self.experience_counts:
            count = self.experience_counts[key]
            initial_c = self._calculate_confidence(count + 1)
            
            # Update existing episode if confidence increased
            if key in self.experience_episodes:
                existing = self.experience_episodes[key]
                if initial_c > existing.c:
                    existing.c = initial_c
                    existing.selection_count += 1
                    logger.debug(
                        f"Repeated experience, confidence increased to {initial_c:.2f}"
                    )
        else:
            # First time experience
            count = 0
            initial_c = 0.5
            self.experience_counts[key] = 1
        
        # Create text description
        if custom_text:
            text = custom_text
        else:
            text = self._format_experience(state, action, result)
        
        # Create or get vector
        if vector is None:
            vector = self._create_vector(state, action, result)
        
        # Create episode
        episode = Episode(
            text=text,
            vec=vector,
            c=initial_c,
            episode_type="experience",
            metadata={
                "state": str(state),
                "action": str(action),
                "result": str(result),
                "repetition_count": count + 1
            }
        )
        
        # Store for future reference
        self.experience_counts[key] = count + 1
        self.experience_episodes[key] = episode
        
        return episode
    
    def _calculate_confidence(self, repetition_count: int) -> float:
        """
        Calculate confidence based on repetition count.
        
        Follows a logarithmic curve approaching 1.0.
        
        Args:
            repetition_count: Number of times experienced
            
        Returns:
            Confidence value
        """
        if repetition_count == 1:
            return 0.5  # First experience
        elif repetition_count == 2:
            return 0.7  # Second experience
        elif repetition_count == 3:
            return 0.85  # Third experience
        else:
            # Asymptotic approach to 0.95
            return min(0.95, 0.7 + 0.1 * np.log(repetition_count))
    
    def _create_key(self, state: Any, action: Any, result: Any) -> str:
        """
        Create unique key for experience.
        
        Args:
            state: State
            action: Action
            result: Result
            
        Returns:
            Unique key string
        """
        # Simple string concatenation
        # In practice, might need more sophisticated hashing
        return f"{state}|{action}|{result}"
    
    def _format_experience(self, state: Any, action: Any, result: Any) -> str:
        """
        Format experience as text.
        
        Args:
            state: State
            action: Action
            result: Result
            
        Returns:
            Formatted text
        """
        return f"State: {state}, Action: {action}, Result: {result}"
    
    def _create_vector(self, state: Any, action: Any, result: Any) -> np.ndarray:
        """
        Create vector representation of experience.
        
        Simple hash-based approach for demonstration.
        Real implementation would use proper encoding.
        
        Args:
            state: State
            action: Action
            result: Result
            
        Returns:
            Vector representation
        """
        # Simple hash-based vector (placeholder)
        key = self._create_key(state, action, result)
        hash_val = hash(key) % (2**32)
        
        # Generate deterministic vector
        np.random.seed(hash_val)
        vector = np.random.normal(0, 1, 384)  # Default dimension
        vector = vector / np.linalg.norm(vector)
        
        return vector.astype(np.float32)
    
    def get_experience_confidence(
        self,
        state: Any,
        action: Any,
        result: Any
    ) -> float:
        """
        Get confidence for a specific experience.
        
        Args:
            state: State
            action: Action
            result: Result
            
        Returns:
            Confidence value (0.0 if not experienced)
        """
        key = self._create_key(state, action, result)
        
        if key in self.experience_episodes:
            return self.experience_episodes[key].c
        else:
            return 0.0
    
    def get_statistics(self) -> dict:
        """
        Get experience tracking statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.experience_counts:
            return {
                "unique_experiences": 0,
                "total_repetitions": 0,
                "avg_repetitions": 0.0,
                "max_repetitions": 0,
                "avg_confidence": 0.0
            }
        
        total_reps = sum(self.experience_counts.values())
        confidences = [ep.c for ep in self.experience_episodes.values()]
        
        return {
            "unique_experiences": len(self.experience_counts),
            "total_repetitions": total_reps,
            "avg_repetitions": total_reps / len(self.experience_counts),
            "max_repetitions": max(self.experience_counts.values()),
            "avg_confidence": np.mean(confidences) if confidences else 0.0
        }
    
    def find_similar_experience(
        self,
        state: Any,
        action: Any,
        threshold: float = 0.8
    ) -> Optional[Tuple[Any, float]]:
        """
        Find similar past experience.
        
        Args:
            state: Current state
            action: Planned action
            threshold: Similarity threshold
            
        Returns:
            (result, confidence) tuple if found, None otherwise
        """
        # Check for exact match first
        for key, episode in self.experience_episodes.items():
            parts = key.split("|")
            if len(parts) >= 3:
                exp_state, exp_action, exp_result = parts[0], parts[1], parts[2]
                
                if str(state) == exp_state and str(action) == exp_action:
                    return exp_result, episode.c
        
        # Could implement fuzzy matching here
        return None