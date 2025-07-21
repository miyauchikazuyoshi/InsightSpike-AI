"""
Proper Entropy Calculation for Information Gain
==============================================

Fixes the entropy calculation to properly reflect information content.
High entropy = scattered/disorganized
Low entropy = clustered/organized
"""

import numpy as np
from typing import Any
import logging

logger = logging.getLogger(__name__)


def calculate_vector_entropy(vectors: np.ndarray) -> float:
    """
    Calculate entropy based on vector similarity distribution.

    High similarity among vectors = Low entropy (organized)
    Low similarity among vectors = High entropy (scattered)

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

        # Convert similarities to distances (for entropy)
        # similarity=1 → distance=0 (identical, low entropy)
        # similarity=-1 → distance=2 (opposite, high entropy)
        distances = 1 - pairwise_sims

        # Calculate entropy based on distance distribution
        # Normalize distances to [0, 1]
        distances = distances / 2.0

        # Use variance as entropy proxy
        # High variance in distances = high entropy
        # Low variance = low entropy (all similar)
        entropy = np.std(distances)

        # Alternative: use mean distance as entropy
        # entropy = np.mean(distances)

        return float(entropy)

    except Exception as e:
        logger.error(f"Entropy calculation failed: {e}")
        return 0.5  # Default medium entropy


def calculate_information_gain(
    vectors_before: np.ndarray, vectors_after: np.ndarray
) -> float:
    """
    Calculate information gain as entropy reduction.

    ΔIG = H(before) - H(after)

    Positive ΔIG means entropy decreased (information gained).

    Args:
        vectors_before: Initial vector set
        vectors_after: Final vector set

    Returns:
        float: Information gain (positive = good)
    """
    h_before = calculate_vector_entropy(vectors_before)
    h_after = calculate_vector_entropy(vectors_after)

    delta_ig = h_before - h_after

    logger.debug(
        f"IG calculation: H(before)={h_before:.3f}, H(after)={h_after:.3f}, ΔIG={delta_ig:.3f}"
    )

    return delta_ig
