"""Factory for creating vector indices with different backends."""

import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np

from .interface import VectorIndexInterface
from .numpy_index import NumpyNearestNeighborIndex, OptimizedNumpyIndex

logger = logging.getLogger(__name__)

# Check FAISS availability
FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logger.info("FAISS not available, using NumPy backend")


class FaissIndexWrapper(VectorIndexInterface):
    """Wrapper for FAISS index to match our interface."""
    
    def __init__(self, dimension: int, index_type: str = "Flat", **kwargs):
        """Initialize FAISS index.
        
        Args:
            dimension: Vector dimension
            index_type: FAISS index type (Flat, IVF, etc.)
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not available")
            
        self.dimension = dimension
        
        if index_type == "Flat":
            # Using IndexFlatIP for inner product (similar to cosine for normalized vectors)
            self.index = faiss.IndexFlatIP(dimension)
        else:
            # Default to flat index
            self.index = faiss.IndexFlatIP(dimension)
            
        self._is_trained = True
        
    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to FAISS index."""
        # FAISS requires float32
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        self.index.add(vectors)
        
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search using FAISS."""
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
        distances, indices = self.index.search(queries, k)
        return distances, indices
    
    def reset(self) -> None:
        """Reset FAISS index."""
        self.index.reset()
        
    @property
    def ntotal(self) -> int:
        """Total vectors in index."""
        return self.index.ntotal
    
    @property
    def is_trained(self) -> bool:
        """FAISS training status."""
        return self._is_trained


class VectorIndexFactory:
    """Factory for creating vector indices."""
    
    @staticmethod
    def create_index(
        dimension: int,
        index_type: str = "auto",
        optimize: bool = True,
        **kwargs
    ) -> VectorIndexInterface:
        """Create a vector index.
        
        Args:
            dimension: Vector dimension
            index_type: Backend type - "faiss", "numpy", or "auto"
            optimize: Whether to use optimized implementations
            **kwargs: Backend-specific parameters
            
        Returns:
            VectorIndexInterface implementation
        """
        # Determine index type
        if index_type == "auto":
            if FAISS_AVAILABLE:
                logger.debug("Auto-selected FAISS backend")
                index_type = "faiss"
            else:
                logger.debug("Auto-selected NumPy backend")
                index_type = "numpy"
                
        # Create index
        if index_type == "faiss":
            if not FAISS_AVAILABLE:
                logger.warning("FAISS requested but not available, falling back to NumPy")
                index_type = "numpy"
            else:
                return FaissIndexWrapper(dimension, **kwargs)
                
        if index_type == "numpy":
            if optimize:
                logger.debug("Creating optimized NumPy index")
                return OptimizedNumpyIndex(dimension, **kwargs)
            else:
                logger.debug("Creating basic NumPy index")
                return NumpyNearestNeighborIndex(dimension, **kwargs)
                
        raise ValueError(f"Unknown index type: {index_type}")
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> VectorIndexInterface:
        """Create index from configuration dictionary.
        
        Args:
            config: Configuration with keys:
                - dimension: int (required)
                - backend: str (optional, default "auto")
                - optimize: bool (optional, default True)
                - Other backend-specific options
                
        Returns:
            VectorIndexInterface implementation
        """
        dimension = config.get("dimension")
        if dimension is None:
            raise ValueError("dimension is required in config")
            
        backend = config.get("backend", "auto")
        optimize = config.get("optimize", True)
        
        # Extract backend-specific options
        backend_options = {
            k: v for k, v in config.items() 
            if k not in ["dimension", "backend", "optimize"]
        }
        
        return VectorIndexFactory.create_index(
            dimension=dimension,
            index_type=backend,
            optimize=optimize,
            **backend_options
        )