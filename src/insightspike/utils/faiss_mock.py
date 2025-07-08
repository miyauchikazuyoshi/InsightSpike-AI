"""
Mock implementation of FAISS for macOS compatibility issues.
This provides a simple numpy-based alternative when FAISS has issues.
"""

import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MockFaissIndex:
    """Simple numpy-based nearest neighbor search as FAISS alternative."""
    
    def __init__(self, dimension: int, index_type: str = "Flat"):
        self.dimension = dimension
        self.index_type = index_type
        self.vectors = None
        self.is_trained = True
        self.ntotal = 0
        
    def add(self, vectors: np.ndarray):
        """Add vectors to index."""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
            
        if self.vectors is None:
            self.vectors = vectors.copy()
        else:
            self.vectors = np.vstack([self.vectors, vectors])
            
        self.ntotal = len(self.vectors)
        
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search k nearest neighbors."""
        if self.vectors is None or self.ntotal == 0:
            # Return empty results
            if queries.ndim == 1:
                return np.array([[-1] * k]), np.array([[0.0] * k])
            else:
                n_queries = queries.shape[0]
                return np.array([[-1] * k] * n_queries), np.array([[0.0] * k] * n_queries)
        
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
            
        # Compute cosine similarity
        queries_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-10)
        vectors_norm = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10)
        
        similarities = np.dot(queries_norm, vectors_norm.T)
        
        # Get top k
        k_actual = min(k, self.ntotal)
        indices = np.argsort(-similarities, axis=1)[:, :k_actual]
        
        distances = np.array([similarities[i, indices[i]] for i in range(len(queries))])
        
        # Pad if needed
        if k_actual < k:
            pad_indices = -np.ones((len(queries), k - k_actual), dtype=np.int64)
            pad_distances = np.zeros((len(queries), k - k_actual))
            indices = np.hstack([indices, pad_indices])
            distances = np.hstack([distances, pad_distances])
            
        return distances.astype(np.float32), indices.astype(np.int64)
        
    def reset(self):
        """Reset index."""
        self.vectors = None
        self.ntotal = 0


def IndexFlatIP(dimension: int) -> MockFaissIndex:
    """Create a mock FAISS IndexFlatIP (inner product index)."""
    return MockFaissIndex(dimension, "FlatIP")


def IndexFlatL2(dimension: int) -> MockFaissIndex:
    """Create a mock FAISS IndexFlatL2 (L2 distance index)."""
    return MockFaissIndex(dimension, "FlatL2")


# Utility function to check if we should use mock
def get_faiss_index(dimension: int, index_type: str = "FlatIP"):
    """Get either real FAISS or mock based on availability."""
    try:
        import faiss
        if index_type == "FlatIP":
            return faiss.IndexFlatIP(dimension)
        else:
            return faiss.IndexFlatL2(dimension)
    except Exception as e:
        logger.warning(f"Using mock FAISS due to: {e}")
        if index_type == "FlatIP":
            return IndexFlatIP(dimension)
        else:
            return IndexFlatL2(dimension)