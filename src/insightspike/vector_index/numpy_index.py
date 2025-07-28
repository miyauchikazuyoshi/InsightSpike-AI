"""NumPy-based vector index implementation."""

from typing import Tuple, Optional
import numpy as np
import logging

from .interface import VectorIndexInterface

logger = logging.getLogger(__name__)


class NumpyNearestNeighborIndex(VectorIndexInterface):
    """NumPy-based nearest neighbor search implementation."""
    
    def __init__(self, dimension: int, **kwargs):
        """Initialize the index.
        
        Args:
            dimension: Vector dimension
            **kwargs: Additional parameters (ignored for compatibility)
        """
        self.dimension = dimension
        self.vectors: Optional[np.ndarray] = None
        self.ids: Optional[np.ndarray] = None
        self._is_trained = True  # Always trained for NumPy
        
    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to the index.
        
        Args:
            vectors: Array of shape (n_vectors, dimension)
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {vectors.shape[1]}")
            
        if self.vectors is None:
            self.vectors = vectors.copy()
            self.ids = np.arange(len(vectors))
        else:
            self.vectors = np.vstack([self.vectors, vectors])
            new_ids = np.arange(len(self.vectors) - len(vectors), len(self.vectors))
            self.ids = np.concatenate([self.ids, new_ids])
            
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors using cosine similarity.
        
        Args:
            queries: Array of shape (n_queries, dimension)
            k: Number of nearest neighbors
            
        Returns:
            distances: Cosine similarities (higher is better)
            indices: Indices of nearest neighbors
        """
        if self.vectors is None or len(self.vectors) == 0:
            # Return empty results if no vectors
            n_queries = len(queries)
            return np.zeros((n_queries, 0)), np.zeros((n_queries, 0), dtype=int)
            
        distances = []
        indices = []
        
        for query in queries:
            # Compute cosine similarities
            similarities = np.dot(self.vectors, query)
            norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query)
            # Avoid division by zero
            similarities = similarities / (norms + 1e-8)
            
            # Get top-k
            actual_k = min(k, len(similarities))
            if actual_k == len(similarities):
                top_k_idx = np.argsort(similarities)[::-1]
            else:
                # Use argpartition for efficiency
                top_k_idx = np.argpartition(similarities, -actual_k)[-actual_k:]
                top_k_idx = top_k_idx[np.argsort(similarities[top_k_idx])[::-1]]
            
            # Pad with zeros if k > number of vectors
            if actual_k < k:
                pad_size = k - actual_k
                top_k_idx = np.pad(top_k_idx, (0, pad_size), constant_values=0)
                similarities_padded = np.pad(
                    similarities[top_k_idx[:actual_k]], 
                    (0, pad_size), 
                    constant_values=0
                )
                distances.append(similarities_padded)
            else:
                distances.append(similarities[top_k_idx])
                
            indices.append(self.ids[top_k_idx[:actual_k]] if actual_k < k 
                          else self.ids[top_k_idx])
            
        return np.array(distances), np.array(indices)
    
    def reset(self) -> None:
        """Clear all vectors from the index."""
        self.vectors = None
        self.ids = None
        
    @property
    def ntotal(self) -> int:
        """Total number of vectors in the index."""
        return 0 if self.vectors is None else len(self.vectors)
    
    @property
    def is_trained(self) -> bool:
        """Always True for NumPy implementation."""
        return self._is_trained


class OptimizedNumpyIndex(NumpyNearestNeighborIndex):
    """Optimized NumPy index with batch processing and caching."""
    
    def __init__(self, dimension: int, normalize: bool = True, **kwargs):
        """Initialize optimized index.
        
        Args:
            dimension: Vector dimension
            normalize: Whether to normalize vectors for faster search
        """
        super().__init__(dimension, **kwargs)
        self.normalize = normalize
        self._normalized_vectors: Optional[np.ndarray] = None
        
    def add(self, vectors: np.ndarray) -> None:
        """Add vectors and invalidate cache."""
        super().add(vectors)
        self._normalized_vectors = None  # Invalidate cache
        
    def _ensure_normalized(self) -> None:
        """Ensure normalized vectors are computed."""
        if self._normalized_vectors is None and self.vectors is not None:
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            self._normalized_vectors = self.vectors / (norms + 1e-8)
            
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized batch search.
        
        Args:
            queries: Array of shape (n_queries, dimension)
            k: Number of nearest neighbors
            
        Returns:
            distances: Cosine similarities
            indices: Indices of nearest neighbors
        """
        if self.vectors is None or len(self.vectors) == 0:
            n_queries = len(queries)
            return np.zeros((n_queries, 0)), np.zeros((n_queries, 0), dtype=int)
            
        # Ensure normalized vectors
        if self.normalize:
            self._ensure_normalized()
            
            # Normalize queries
            query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
            normalized_queries = queries / (query_norms + 1e-8)
            
            # Batch matrix multiplication
            similarities = np.dot(normalized_queries, self._normalized_vectors.T)
        else:
            # Direct computation without normalization
            similarities = np.dot(queries, self.vectors.T)
            
        # Process each query's results
        n_vectors = len(self.vectors)
        actual_k = min(k, n_vectors)
        
        distances = np.zeros((len(queries), k))
        indices = np.zeros((len(queries), k), dtype=int)
        
        for i, sim in enumerate(similarities):
            if actual_k == n_vectors:
                top_k_idx = np.argsort(sim)[::-1]
            else:
                # Efficient top-k with argpartition
                top_k_idx = np.argpartition(sim, -actual_k)[-actual_k:]
                top_k_idx = top_k_idx[np.argsort(sim[top_k_idx])[::-1]]
                
            # Fill results (pad with zeros if needed)
            distances[i, :actual_k] = sim[top_k_idx]
            indices[i, :actual_k] = self.ids[top_k_idx]
            
        return distances, indices
    
    def reset(self) -> None:
        """Clear index and cache."""
        super().reset()
        self._normalized_vectors = None