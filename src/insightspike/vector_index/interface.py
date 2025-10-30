"""Vector index interface definition."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
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
    
    def radius_search(self, queries: np.ndarray, radius: float, 
                      max_results: Optional[int] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Search for all vectors within a given radius/distance.
        
        Args:
            queries: Array of shape (n_queries, dimension)
            radius: Maximum distance/minimum similarity threshold
            max_results: Optional maximum number of results per query
            
        Returns:
            distances: List of arrays with distances/similarities
            indices: List of arrays with indices
            
        Note: Default implementation uses search with large k and filters.
        Subclasses can override for more efficient implementation.
        """
        # Default implementation: search with large k and filter
        k = min(self.ntotal, max_results) if max_results else self.ntotal
        if k == 0:
            return [], []
        
        distances, indices = self.search(queries, k)
        
        # Filter by radius (assuming similarity for normalize=True, distance for normalize=False)
        filtered_results = []
        filtered_indices = []
        
        for i in range(len(queries)):
            mask = distances[i] <= radius if not hasattr(self, 'normalize') or not self.normalize else distances[i] >= radius
            filtered_results.append(distances[i][mask])
            filtered_indices.append(indices[i][mask])
        
        return filtered_results, filtered_indices
    
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