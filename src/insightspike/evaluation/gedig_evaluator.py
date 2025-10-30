"""
geDIG (Graph Edit Distance - Information Gain) Evaluator
========================================================

Evaluates episode combinations based on structural novelty and information gain.
C-values are NOT used in evaluation to preserve accuracy, only updated after selection.
"""

import logging
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np

from ..core.episode import Episode

logger = logging.getLogger(__name__)


class GeDIGEvaluator:
    """
    Evaluates episode combinations using geDIG metric.
    
    geDIG = GED - k * IG
    - GED: Graph Edit Distance (structural novelty)
    - IG: Information Gain (variance/deviation)
    - k: Balance coefficient
    
    C-values are NOT used in calculation to preserve IG accuracy.
    They are only updated after selection as validation feedback.
    """
    
    def __init__(self, k: float = 0.5, combination_size: int = 3):
        """
        Initialize geDIG evaluator.
        
        Args:
            k: Balance coefficient between GED and IG (default 0.5)
            combination_size: Number of episodes to combine (default 3)
        """
        self.k = k
        self.combination_size = combination_size
    
    def evaluate(self, episode_combination: List[Episode]) -> float:
        """
        Evaluate a combination of episodes using geDIG.
        
        Pure structural evaluation without C-value influence.
        
        Args:
            episode_combination: List of episodes to evaluate
            
        Returns:
            geDIG score (lower is better)
        """
        # Calculate Graph Edit Distance
        ged = self._calculate_ged(episode_combination)
        
        # Calculate Information Gain (pure variance calculation)
        ig = self._calculate_information_gain(episode_combination)
        
        # geDIG value (lower is better)
        gedig_value = ged - self.k * ig
        
        return gedig_value
    
    def rerank(
        self, 
        candidates: List[Episode], 
        top_k: int = 10
    ) -> Tuple[List[Episode], List[Episode]]:
        """
        Rerank candidate episodes using geDIG evaluation.
        
        Returns both selected and non-selected episodes for external C-value updates.
        
        Args:
            candidates: List of candidate episodes
            top_k: Number of top episodes to return
            
        Returns:
            Tuple of (selected_episodes, all_candidates)
        """
        if len(candidates) <= self.combination_size:
            # Not enough candidates for combination
            return candidates, candidates
        
        # Generate all possible combinations
        all_combinations = list(combinations(candidates, self.combination_size))
        
        # Evaluate each combination
        scored_combinations = []
        for combo in all_combinations:
            score = self.evaluate(list(combo))
            scored_combinations.append((list(combo), score))
        
        # Sort by geDIG score (lower is better)
        scored_combinations.sort(key=lambda x: x[1])
        
        # Get best combination
        best_combination = scored_combinations[0][0]
        
        # Return selected and all candidates for external processing
        return best_combination, candidates
    
    def _calculate_ged(self, episodes: List[Episode]) -> float:
        """
        Calculate Graph Edit Distance (structural novelty).
        
        Args:
            episodes: List of episodes
            
        Returns:
            GED score
        """
        if len(episodes) < 2:
            return 0.0
        
        # Calculate pairwise distances between episode vectors
        total_distance = 0.0
        count = 0
        
        for i in range(len(episodes)):
            for j in range(i + 1, len(episodes)):
                # Euclidean distance between vectors
                distance = np.linalg.norm(episodes[i].vec - episodes[j].vec)
                total_distance += distance
                count += 1
        
        # Average distance as proxy for GED
        return total_distance / count if count > 0 else 0.0
    
    def _calculate_information_gain(self, episodes: List[Episode]) -> float:
        """
        Calculate Information Gain based on variance.
        
        Pure calculation without C-value influence to maintain accuracy.
        
        Args:
            episodes: List of episodes
            
        Returns:
            Information Gain score
        """
        if not episodes:
            return 0.0
        
        # Stack vectors for variance calculation
        vectors = np.array([ep.vec for ep in episodes])
        
        # Calculate variance across dimensions
        if len(vectors) > 1:
            # Variance of the combination
            variance = np.var(vectors, axis=0)
            # Total information gain
            ig = np.sum(variance)
        else:
            # Single episode has no variance
            ig = 0.0
        
        return ig
    
    def evaluate_with_scores(
        self,
        candidates: List[Episode]
    ) -> List[Tuple[List[Episode], float]]:
        """
        Evaluate all combinations and return with scores.
        
        Pure evaluation without side effects.
        
        Args:
            candidates: List of candidate episodes
            
        Returns:
            List of (combination, score) tuples sorted by score
        """
        combinations = self.generate_combinations(candidates)
        
        results = []
        for combo in combinations:
            score = self.evaluate(combo)
            results.append((combo, score))
        
        # Sort by score (lower is better)
        results.sort(key=lambda x: x[1])
        
        return results
    
    def generate_combinations(
        self,
        episodes: List[Episode],
        size: Optional[int] = None
    ) -> List[List[Episode]]:
        """
        Generate episode combinations for evaluation.
        
        Args:
            episodes: List of episodes
            size: Combination size (defaults to self.combination_size)
            
        Returns:
            List of episode combinations
        """
        combo_size = size or self.combination_size
        
        if len(episodes) <= combo_size:
            return [episodes]
        
        return [list(combo) for combo in combinations(episodes, combo_size)]
    
    def evaluate_all(
        self,
        episodes: List[Episode],
        size: Optional[int] = None
    ) -> List[Tuple[List[Episode], float]]:
        """
        Evaluate all possible combinations.
        
        Args:
            episodes: List of episodes
            size: Combination size
            
        Returns:
            List of (combination, score) tuples sorted by score
        """
        combinations = self.generate_combinations(episodes, size)
        
        results = []
        for combo in combinations:
            score = self.evaluate(combo)
            results.append((combo, score))
        
        # Sort by score (lower is better)
        results.sort(key=lambda x: x[1])
        
        return results