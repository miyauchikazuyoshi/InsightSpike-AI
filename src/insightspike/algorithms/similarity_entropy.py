"""
Simple and reliable entropy calculation based on vector similarity
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_similarity_entropy(vectors: np.ndarray) -> float:
    """
    Calculate entropy based on pairwise similarities.
    
    High average similarity = Low entropy (organized)
    Low average similarity = High entropy (scattered)
    
    Args:
        vectors: Array of shape (n_samples, n_features)
        
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
        
        # Convert to entropy: high similarity = low entropy
        # Map similarity [-1, 1] to entropy [1, 0]
        entropy = (1 - avg_similarity) / 2
        
        return float(entropy)
        
    except Exception as e:
        logger.error(f"Entropy calculation failed: {e}")
        return 0.5  # Default medium entropy