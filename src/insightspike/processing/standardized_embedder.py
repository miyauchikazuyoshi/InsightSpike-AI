"""
Standardized Embedder - Ensures consistent embedding shapes across the system
"""

import numpy as np
from typing import Union, List
import logging

logger = logging.getLogger(__name__)


class StandardizedEmbedder:
    """
    Wrapper that ensures embeddings are always returned as 1D arrays of shape (dim,).
    
    This solves the inconsistency where some embedders return (1, dim) while
    the system expects (dim,).
    """
    
    def __init__(self, base_embedder):
        """
        Args:
            base_embedder: Any embedder with get_embedding() or embed() method
        """
        self.base_embedder = base_embedder
        
        # Detect which method to use
        if hasattr(base_embedder, 'get_embedding'):
            self._embed_method = 'get_embedding'
        elif hasattr(base_embedder, 'embed'):
            self._embed_method = 'embed'
        elif hasattr(base_embedder, 'encode'):
            self._embed_method = 'encode'
        else:
            raise ValueError("Base embedder must have get_embedding(), embed(), or encode() method")
        
        logger.info(f"StandardizedEmbedder initialized with {type(base_embedder).__name__} using {self._embed_method}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text, always returning shape (dim,).
        
        Args:
            text: Text to embed
            
        Returns:
            np.ndarray of shape (dim,), typically (384,)
        """
        # Call appropriate method
        if self._embed_method == 'get_embedding':
            embedding = self.base_embedder.get_embedding(text)
        elif self._embed_method == 'embed':
            embedding = self.base_embedder.embed(text)
        else:  # encode
            embedding = self.base_embedder.encode(text)
        
        # Ensure numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        # Fix shape
        original_shape = embedding.shape
        
        # Handle various cases
        if embedding.ndim == 1:
            # Already 1D, good
            pass
        elif embedding.ndim == 2 and embedding.shape[0] == 1:
            # Shape (1, dim) -> (dim,)
            embedding = embedding.squeeze(0)
        elif embedding.ndim == 2 and embedding.shape[1] == 1:
            # Shape (dim, 1) -> (dim,)
            embedding = embedding.squeeze(1)
        else:
            # Unexpected shape, try to flatten
            logger.warning(f"Unexpected embedding shape {original_shape}, flattening")
            embedding = embedding.flatten()
        
        if original_shape != embedding.shape:
            logger.debug(f"Standardized embedding shape from {original_shape} to {embedding.shape}")
        
        return embedding
    
    def embed(self, text: str) -> np.ndarray:
        """Alias for get_embedding for compatibility."""
        return self.get_embedding(text)
    
    def encode(self, text: str) -> np.ndarray:
        """Alias for get_embedding for compatibility."""
        return self.get_embedding(text)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple texts, returning shape (n, dim).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            np.ndarray of shape (n, dim)
        """
        embeddings = [self.get_embedding(text) for text in texts]
        return np.vstack(embeddings)
    
    # Forward other attributes to base embedder
    def __getattr__(self, name):
        return getattr(self.base_embedder, name)