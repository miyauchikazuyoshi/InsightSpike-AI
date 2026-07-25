
"""
Flash-geDIG: High-Velocity Structural Inference Metrics for Transformers.

"Structure as a First-Class Citizen in Deep Learning"

The canonical geDIG API compares before/after states and follows the
repository-wide ``lower F is better`` convention. The structural-profile API
is the historical single-state Flash diagnostic.

Usage:
    >>> import torch
    >>> from insightspike.gedig import compute_delta_f_score
    >>> before = torch.rand(1, 12, 64, 64)
    >>> after = torch.rand(1, 12, 64, 64)
    >>> result = compute_delta_f_score(before, after)
    >>> print(result.F_mean)
"""

from .functional import (
    compute_delta_f_score,
    compute_f_score,
    compute_structural_profile,
)
from .module import FlashGeDIGLoss, GeDIGObserver

__all__ = [
    "compute_delta_f_score",
    "compute_f_score",
    "compute_structural_profile",
    "FlashGeDIGLoss",
    "GeDIGObserver",
]
