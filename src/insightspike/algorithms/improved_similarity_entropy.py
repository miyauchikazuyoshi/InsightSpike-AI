"""
Improved entropy calculation with better normalization
"""

import numpy as np
import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class NormalizationMethod(Enum):
    """Available normalization methods"""
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    EXPONENTIAL = "exponential"
    PIECEWISE = "piecewise"


def calculate_similarity_entropy(
    vectors: np.ndarray,
    method: NormalizationMethod = NormalizationMethod.SIGMOID,
    **kwargs
) -> float:
    """
    Calculate entropy based on pairwise similarities with improved normalization.
    
    Args:
        vectors: Array of shape (n_samples, n_features)
        method: Normalization method to use
        **kwargs: Additional parameters for normalization
        
    Returns:
        float: Entropy value between 0 and 1
    """
    if vectors is None or len(vectors) < 2:
        return 0.0
        
    try:
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / (norms + 1e-8)
        
        # Calculate pairwise similarities
        similarities = np.dot(normalized, normalized.T)
        
        # Get upper triangle (excluding diagonal)
        n = len(vectors)
        upper_indices = np.triu_indices(n, k=1)
        pairwise_sims = similarities[upper_indices]
        
        # Average similarity
        avg_similarity = np.mean(pairwise_sims)
        
        # Apply normalization
        if method == NormalizationMethod.LINEAR:
            entropy = linear_normalization(avg_similarity)
        elif method == NormalizationMethod.SIGMOID:
            steepness = kwargs.get('steepness', 5.0)
            entropy = sigmoid_normalization(avg_similarity, steepness)
        elif method == NormalizationMethod.EXPONENTIAL:
            alpha = kwargs.get('alpha', 2.0)
            entropy = exponential_normalization(avg_similarity, alpha)
        elif method == NormalizationMethod.PIECEWISE:
            entropy = piecewise_normalization(avg_similarity)
        else:
            # Default to linear
            entropy = linear_normalization(avg_similarity)
        
        return float(entropy)
        
    except Exception as e:
        logger.error(f"Entropy calculation failed: {e}")
        return 0.5  # Default medium entropy


def linear_normalization(similarity: float) -> float:
    """
    Linear mapping from similarity to entropy.
    
    similarity: [-1, 1] -> entropy: [1, 0]
    """
    return (1 - similarity) / 2


def sigmoid_normalization(similarity: float, steepness: float = 5.0) -> float:
    """
    Sigmoid-based normalization for smoother transitions.
    More sensitive near high similarities.
    
    Args:
        similarity: Average similarity value [-1, 1]
        steepness: Controls the steepness of the sigmoid (higher = steeper)
    """
    # Map similarity to sigmoid input
    x = -steepness * similarity
    return 1 / (1 + np.exp(-x))


def exponential_normalization(similarity: float, alpha: float = 2.0) -> float:
    """
    Exponential normalization emphasizing high similarities.
    
    Args:
        similarity: Average similarity value [-1, 1]
        alpha: Decay parameter (higher = faster decay)
    """
    # Map to [0, 1] first
    normalized_sim = (similarity + 1) / 2
    # Apply exponential decay
    return 1 - np.exp(-alpha * (1 - normalized_sim))


def piecewise_normalization(similarity: float) -> float:
    """
    Custom piecewise normalization with different sensitivity regions.
    
    Regions:
    - Very high similarity (>0.8): Very low entropy, high sensitivity
    - High similarity (0.5-0.8): Low entropy, medium sensitivity
    - Medium similarity (-0.5-0.5): Moderate entropy
    - Low similarity (<-0.5): High entropy
    """
    if similarity > 0.8:
        # Very high similarity: entropy 0.0-0.1
        return 0.1 * (1 - similarity) / 0.2
    elif similarity > 0.5:
        # High similarity: entropy 0.1-0.3
        return 0.1 + 0.2 * (0.8 - similarity) / 0.3
    elif similarity > -0.5:
        # Medium similarity: entropy 0.3-0.7
        return 0.3 + 0.4 * (0.5 - similarity) / 1.0
    else:
        # Low similarity: entropy 0.7-1.0
        return 0.7 + 0.3 * (-0.5 - similarity) / 0.5


def calculate_entropy_change(
    before_vectors: np.ndarray,
    after_vectors: np.ndarray,
    method: NormalizationMethod = NormalizationMethod.SIGMOID,
    **kwargs
) -> float:
    """
    Calculate the change in entropy (ΔH).
    
    Positive value means entropy increased (more disorder).
    Negative value means entropy decreased (more organization).
    
    Args:
        before_vectors: Vectors before the change
        after_vectors: Vectors after the change
        method: Normalization method
        **kwargs: Additional parameters
        
    Returns:
        float: ΔH (entropy_after - entropy_before)
    """
    entropy_before = calculate_similarity_entropy(before_vectors, method, **kwargs)
    entropy_after = calculate_similarity_entropy(after_vectors, method, **kwargs)
    
    return entropy_after - entropy_before


def calculate_information_gain(
    before_vectors: np.ndarray,
    after_vectors: np.ndarray,
    method: NormalizationMethod = NormalizationMethod.SIGMOID,
    **kwargs
) -> float:
    """
    Calculate information gain (IG).
    
    IG = H(before) - H(after)
    
    Positive value means information was gained (entropy decreased).
    Negative value means information was lost (entropy increased).
    
    Args:
        before_vectors: Vectors before the change
        after_vectors: Vectors after the change
        method: Normalization method
        **kwargs: Additional parameters
        
    Returns:
        float: Information gain
    """
    return -calculate_entropy_change(before_vectors, after_vectors, method, **kwargs)