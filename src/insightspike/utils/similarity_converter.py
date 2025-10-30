"""
Utility to convert between distance and cosine similarity metrics
"""

import numpy as np
from typing import Union, Tuple


class SimilarityConverter:
    """Convert between distance and cosine similarity for normalized vectors."""
    
    @staticmethod
    def distance_to_cosine(distance: float) -> float:
        """
        Convert Euclidean distance to cosine similarity for normalized vectors.
        
        For normalized vectors: ||a - b||² = 2 - 2·cos(a,b)
        Therefore: cos(a,b) = 1 - (distance²/2)
        
        Args:
            distance: Euclidean distance between normalized vectors
            
        Returns:
            Cosine similarity in range [-1, 1]
        """
        # Clamp distance to valid range [0, 2]
        distance = np.clip(distance, 0, 2)
        return 1 - (distance ** 2) / 2
    
    @staticmethod
    def cosine_to_distance(cosine_sim: float) -> float:
        """
        Convert cosine similarity to Euclidean distance for normalized vectors.
        
        Args:
            cosine_sim: Cosine similarity in range [-1, 1]
            
        Returns:
            Euclidean distance in range [0, 2]
        """
        # Clamp cosine similarity to valid range [-1, 1]
        cosine_sim = np.clip(cosine_sim, -1, 1)
        return np.sqrt(2 - 2 * cosine_sim)
    
    @staticmethod
    def distance_to_relevance(distance: float, reference_distance: float = 1.0) -> float:
        """
        Convert distance to a relevance score (0-1) for LLM prompts.
        
        Uses exponential decay from reference distance.
        
        Args:
            distance: Euclidean distance
            reference_distance: Distance at which relevance = 0.5
            
        Returns:
            Relevance score in range [0, 1]
        """
        # Exponential decay: relevance = exp(-k * distance)
        # At reference_distance, relevance should be 0.5
        # So: 0.5 = exp(-k * ref_dist) => k = ln(2) / ref_dist
        k = np.log(2) / reference_distance
        return np.exp(-k * distance)
    
    @staticmethod
    def cosine_to_angle_degrees(cosine_sim: float) -> float:
        """
        Convert cosine similarity to angle in degrees.
        
        Args:
            cosine_sim: Cosine similarity value [-1, 1]
            
        Returns:
            Angle in degrees [0, 180]
        """
        # Clamp to valid range to handle numerical errors
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        angle_radians = np.arccos(cosine_sim)
        return np.degrees(angle_radians)
    
    @staticmethod
    def format_for_llm(distance: float, cosine_sim: float = None) -> str:
        """
        Format metrics for LLM understanding.
        
        Simply provide both metrics - modern LLMs understand these values.
        Now includes angle in degrees for better intuition.
        
        Args:
            distance: Euclidean distance
            cosine_sim: Optional pre-computed cosine similarity
            
        Returns:
            Formatted metric string
        """
        if cosine_sim is None:
            cosine_sim = SimilarityConverter.distance_to_cosine(distance)
        
        angle_degrees = SimilarityConverter.cosine_to_angle_degrees(cosine_sim)
        
        # Include angle for better intuition
        return f"dist={distance:.3f}, angle={angle_degrees:.1f}°, cos={cosine_sim:.3f}"
    
    @staticmethod
    def format_metrics(distance: float, cosine_sim: float = None) -> str:
        """
        Format both metrics for display.
        
        Args:
            distance: Euclidean distance
            cosine_sim: Optional pre-computed cosine similarity
            
        Returns:
            Formatted string with both metrics
        """
        if cosine_sim is None:
            cosine_sim = SimilarityConverter.distance_to_cosine(distance)
        
        angle_degrees = SimilarityConverter.cosine_to_angle_degrees(cosine_sim)
        
        return f"Distance: {distance:.3f}, Angle: {angle_degrees:.1f}°, Cosine: {cosine_sim:.3f}"
    
    @staticmethod
    def get_both_metrics(vec1: np.ndarray, vec2: np.ndarray) -> Tuple[float, float]:
        """
        Calculate both distance and cosine similarity between vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Tuple of (distance, cosine_similarity)
        """
        # Normalize if not already normalized
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        
        # Calculate metrics
        distance = np.linalg.norm(vec1_norm - vec2_norm)
        cosine_sim = np.dot(vec1_norm, vec2_norm)
        
        return distance, cosine_sim