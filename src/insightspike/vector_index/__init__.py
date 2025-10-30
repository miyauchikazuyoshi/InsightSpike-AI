"""Vector index implementations for InsightSpike-AI.

This module provides vector similarity search functionality with multiple backends.
"""

from .interface import VectorIndexInterface
from .numpy_index import NumpyNearestNeighborIndex, OptimizedNumpyIndex
from .factory import VectorIndexFactory

__all__ = [
    "VectorIndexInterface",
    "NumpyNearestNeighborIndex", 
    "OptimizedNumpyIndex",
    "VectorIndexFactory",
]