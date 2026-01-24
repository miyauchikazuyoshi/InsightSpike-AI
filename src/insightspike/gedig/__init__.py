
"""
Flash-geDIG: High-Velocity Structural Inference Metrics for Transformers.

"Structure as a First-Class Citizen in Deep Learning"

Usage:
    >>> import torch
    >>> from insightspike.gedig import compute_f_score
    >>> attn = torch.rand(1, 12, 64, 64)
    >>> f, metrics = compute_f_score(attn)
    >>> print(f.mean())
"""

from .functional import compute_f_score
from .module import FlashGeDIGLoss, GeDIGObserver

__all__ = [
    "compute_f_score",
    "FlashGeDIGLoss",
    "GeDIGObserver",
]
