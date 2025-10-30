"""Simple Top-K Calculator (placeholder for tests).

Provides a deterministic selection of top-k items by score for adaptive processor tests.
"""
from __future__ import annotations
from typing import List, Sequence, Any

class SimpleTopKCalculator:
    def __init__(self, k: int = 5):
        self.k = k

    def select(self, scores: Sequence[float], items: Sequence[Any] | None = None):
        if items is None:
            items = list(range(len(scores)))
        paired = list(zip(scores, items))
        # Sort descending by score
        paired.sort(key=lambda x: x[0], reverse=True)
        return paired[: self.k]

__all__ = ["SimpleTopKCalculator"]
