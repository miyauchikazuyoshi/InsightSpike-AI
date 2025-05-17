"""L1 – Uncertainty metric"""
import numpy as np
from typing import Sequence

__all__ = ["uncertainty"]

def uncertainty(scores: Sequence[float]) -> float:
    probs = np.array(scores, dtype=float)
    probs = probs / (probs.sum() + 1e-9)
    return float(-np.sum(probs * np.log(probs + 1e-9)))