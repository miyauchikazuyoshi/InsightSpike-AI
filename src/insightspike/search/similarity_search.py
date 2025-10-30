"""
Similarity Search with Weight Vector Support
============================================

Provides similarity search that uses weight vectors but not C-values.
C-values are only used post-selection for confidence updates.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..core.episode import Episode
from ..core.weight_vector_manager import WeightVectorManager

logger = logging.getLogger(__name__)


class SimilaritySearch:
    """
    Similarity search that applies weight vectors but ignores C-values.
    
    The search process:
    1. Apply weight vectors to adjust task-specific dimensions
    2. Calculate pure cosine similarity
    3. Return top-k results without C-value influence
    """
    
    def __init__(self, weight_manager: Optional[WeightVectorManager] = None):
        """
        Initialize similarity search.
        
        Args:
            weight_manager: Optional weight manager for dimension adjustment
        """
        self.weight_manager = weight_manager
    
    def search(
        self, 
        query_vec: np.ndarray, 
        episodes: List[Episode], 
        k: int = 30
    ) -> List[Episode]:
        """
        Search for similar episodes using weighted cosine similarity.
        
        C-values are NOT used in search to maintain pure similarity calculation.
        
        Args:
            query_vec: Query vector
            episodes: List of episodes to search
            k: Number of results to return
            
        Returns:
            Top-k most similar episodes
        """
        if not episodes:
            return []
        
        # Apply weights to query if weight manager is available
        if self.weight_manager and self.weight_manager.is_enabled():
            weighted_query = self.weight_manager.apply_weights(query_vec)
        else:
            weighted_query = query_vec
        
        # Calculate similarities
        similarities = []
        for episode in episodes:
            # Apply weights to episode vector
            if self.weight_manager and self.weight_manager.is_enabled():
                weighted_ep = self.weight_manager.apply_weights(episode.vec)
            else:
                weighted_ep = episode.vec
            
            # Pure cosine similarity (no C-value influence)
            sim = self._cosine_similarity(weighted_query, weighted_ep)
            similarities.append((episode, sim))
        
        # Sort by similarity only
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k episodes
        return [ep for ep, _ in similarities[:k]]
    
    def search_with_scores(
        self,
        query_vec: np.ndarray,
        episodes: List[Episode],
        k: int = 30
    ) -> List[Tuple[Episode, float]]:
        """
        Search and return episodes with their similarity scores.
        
        Args:
            query_vec: Query vector
            episodes: List of episodes to search
            k: Number of results to return
            
        Returns:
            List of (episode, similarity_score) tuples
        """
        if not episodes:
            return []
        
        # Apply weights to query
        if self.weight_manager and self.weight_manager.is_enabled():
            weighted_query = self.weight_manager.apply_weights(query_vec)
        else:
            weighted_query = query_vec
        
        # Calculate similarities
        similarities = []
        for episode in episodes:
            # Apply weights to episode
            if self.weight_manager and self.weight_manager.is_enabled():
                weighted_ep = self.weight_manager.apply_weights(episode.vec)
            else:
                weighted_ep = episode.vec
            
            # Pure similarity
            sim = self._cosine_similarity(weighted_query, weighted_ep)
            similarities.append((episode, sim))
        
        # Sort and return top-k with scores
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        # Handle edge cases
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def batch_search(
        self,
        query_vecs: np.ndarray,
        episodes: List[Episode],
        k: int = 30
    ) -> List[List[Episode]]:
        """
        Batch search for multiple queries.
        
        Args:
            query_vecs: Array of query vectors
            episodes: List of episodes to search
            k: Number of results per query
            
        Returns:
            List of result lists for each query
        """
        results = []
        for query_vec in query_vecs:
            results.append(self.search(query_vec, episodes, k))
        return results