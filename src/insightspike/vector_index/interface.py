"""Vector index interface definition."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


class VectorIndexInterface(ABC):
    """Abstract base class for vector index implementations."""
    
    @abstractmethod
    def __init__(self, dimension: int, **kwargs):
        """Initialize the index with given dimension.
        
        Args:
            dimension: Vector dimension
            **kwargs: Implementation-specific parameters
        """
        pass
    
    @abstractmethod
    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to the index.
        
        Args:
            vectors: Array of shape (n_vectors, dimension)
        """
        pass
    
    @abstractmethod
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors.
        
        Args:
            queries: Array of shape (n_queries, dimension)
            k: Number of nearest neighbors to return
            
        Returns:
            distances: Array of shape (n_queries, k) with distances/similarities
            indices: Array of shape (n_queries, k) with indices
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Clear all vectors from the index."""
        pass
    
    @property
    @abstractmethod
    def ntotal(self) -> int:
        """Total number of vectors in the index."""
        pass
    
    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Whether the index is trained (for compatibility with FAISS)."""
        pass