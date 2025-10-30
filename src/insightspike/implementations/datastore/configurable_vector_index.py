"""Configurable vector index implementation using VectorIndexFactory."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ...core.base.datastore import VectorIndex
from ...vector_index import VectorIndexFactory

logger = logging.getLogger(__name__)


class ConfigurableVectorIndex(VectorIndex):
    """Vector index implementation using configurable backend."""

    def __init__(self, dimension: int, index_type: str = "auto", **kwargs):
        """Initialize vector index.
        
        Args:
            dimension: Vector dimension
            index_type: Backend type - "faiss", "numpy", or "auto"
            **kwargs: Additional backend-specific parameters
        """
        self.dimension = dimension
        self.index_type = index_type
        
        # Create the underlying index
        self.index = VectorIndexFactory.create_index(
            dimension=dimension,
            index_type=index_type,
            **kwargs
        )
        
        # ID mapping for compatibility with the VectorIndex interface
        self.id_map = {}  # Map from internal ID to external ID
        self.reverse_id_map = {}  # Map from external ID to internal ID
        self._next_internal_id = 0

    def add_vectors(self, vectors: np.ndarray, ids: Optional[List[int]] = None) -> bool:
        """Add vectors to index.
        
        Args:
            vectors: Array of shape (n_vectors, dimension)
            ids: Optional list of IDs for the vectors
            
        Returns:
            True if successful
        """
        try:
            if ids is None:
                ids = list(range(len(vectors)))
                
            # Add to underlying index
            start_idx = self.index.ntotal
            self.index.add(vectors.astype(np.float32))
            
            # Update ID mappings
            for i, external_id in enumerate(ids):
                internal_id = start_idx + i
                self.id_map[internal_id] = external_id
                self.reverse_id_map[external_id] = internal_id
                
            return True
        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            return False

    def search(
        self, query_vectors: np.ndarray, k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbors.
        
        Args:
            query_vectors: Array of shape (n_queries, dimension)
            k: Number of neighbors to find
            
        Returns:
            distances: Array of shape (n_queries, k)
            indices: Array of shape (n_queries, k) with external IDs
        """
        # Search using underlying index
        distances, internal_indices = self.index.search(
            query_vectors.astype(np.float32), k
        )
        
        # Map internal indices to external IDs
        external_indices = np.zeros_like(internal_indices)
        for i in range(internal_indices.shape[0]):
            for j in range(internal_indices.shape[1]):
                internal_id = internal_indices[i, j]
                if internal_id >= 0 and internal_id in self.id_map:
                    external_indices[i, j] = self.id_map[internal_id]
                else:
                    external_indices[i, j] = -1  # Invalid
                    
        return distances, external_indices

    def remove_vectors(self, ids: List[int]) -> bool:
        """Remove vectors from index.
        
        Note: This requires rebuilding the index as most backends
        don't support efficient removal.
        
        Args:
            ids: List of external IDs to remove
            
        Returns:
            True if successful
        """
        logger.warning("Vector removal requires index rebuild - this may be slow")
        
        try:
            # Get all current vectors and IDs
            if self.index.ntotal == 0:
                return True
                
            # This is a simplified approach - in practice, we'd need to store
            # the original vectors or rebuild from the data source
            logger.warning("Vector removal not fully implemented - index rebuild required")
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove vectors: {e}")
            return False

    def save_index(self, path: str) -> bool:
        """Save index to disk.
        
        Args:
            path: Path to save the index
            
        Returns:
            True if successful
        """
        try:
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Save index metadata
            metadata = {
                "dimension": self.dimension,
                "index_type": self.index_type,
                "id_map": self.id_map,
                "reverse_id_map": self.reverse_id_map,
                "ntotal": self.index.ntotal
            }
            
            metadata_path = path_obj.with_suffix('.meta.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
                
            # For NumPy backend, we need to save the vectors
            # FAISS has its own save mechanism
            if hasattr(self.index, 'vectors') and self.index.vectors is not None:
                vectors_path = path_obj.with_suffix('.npy')
                np.save(vectors_path, self.index.vectors)
                
            logger.info(f"Saved index to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False

    def load_index(self, path: str) -> bool:
        """Load index from disk.
        
        Args:
            path: Path to load the index from
            
        Returns:
            True if successful
        """
        try:
            path_obj = Path(path)
            
            # Load metadata
            metadata_path = path_obj.with_suffix('.meta.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            # Restore ID mappings
            self.id_map = {int(k): v for k, v in metadata['id_map'].items()}
            self.reverse_id_map = {int(k): int(v) for k, v in metadata['reverse_id_map'].items()}
            
            # For NumPy backend, load vectors
            if self.index_type == "numpy" or metadata['index_type'] == "numpy":
                vectors_path = path_obj.with_suffix('.npy')
                if vectors_path.exists():
                    vectors = np.load(vectors_path)
                    self.index.reset()
                    self.index.add(vectors)
                    
            logger.info(f"Loaded index from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    @property
    def ntotal(self) -> int:
        """Total number of vectors in the index."""
        return self.index.ntotal

    def clear(self) -> bool:
        """Clear all vectors from the index.
        
        Returns:
            True if successful
        """
        try:
            self.index.reset()
            self.id_map.clear()
            self.reverse_id_map.clear()
            return True
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            return False