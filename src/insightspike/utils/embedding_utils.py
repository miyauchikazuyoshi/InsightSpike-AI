"""
Embedding Utilities
==================

Utilities for handling embedding shapes and normalization.

DIAG: Emits import start/end markers when INSIGHTSPIKE_DIAG_IMPORT=1 to help
trace stalls during layered agent import.
"""

import os as _os
_EMB_UTIL_DIAG = _os.getenv('INSIGHTSPIKE_DIAG_IMPORT') == '1'
if _EMB_UTIL_DIAG:
    print('[embedding_utils] module import start', flush=True)

import numpy as np
from typing import Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def normalize_embedding_shape(
    embedding: Union[np.ndarray, 'torch.Tensor', List[float]]
) -> np.ndarray:
    """
    Normalize embedding to shape (dim,) instead of (1, dim).
    
    Args:
        embedding: Input embedding in various formats
        
    Returns:
        np.ndarray with shape (dim,)
    """
    # Convert to numpy if needed
    try:
        import torch
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        elif isinstance(embedding, list):
            embedding = np.array(embedding)
    except ImportError:
        if isinstance(embedding, list):
            embedding = np.array(embedding)
    
    # Ensure it's numpy array
    if not isinstance(embedding, np.ndarray):
        raise ValueError(f"Cannot normalize embedding of type {type(embedding)}")
    
    # Handle various shapes
    if embedding.ndim == 0:
        # Scalar, shouldn't happen for embeddings
        raise ValueError("Embedding cannot be a scalar")
    elif embedding.ndim == 1:
        # Already correct shape
        return embedding
    elif embedding.ndim == 2:
        # Shape (1, dim) -> (dim,)
        if embedding.shape[0] == 1:
            return embedding.squeeze(0)
        # Shape (dim, 1) -> (dim,)
        elif embedding.shape[1] == 1:
            return embedding.squeeze(1)
        else:
            raise ValueError(f"Cannot normalize 2D embedding with shape {embedding.shape}")
    else:
        # Higher dimensions
        raise ValueError(f"Cannot normalize embedding with {embedding.ndim} dimensions")


def normalize_batch_embeddings(
    embeddings: Union[np.ndarray, 'torch.Tensor', List]
) -> np.ndarray:
    """
    Normalize a batch of embeddings to shape (batch_size, dim).
    
    Args:
        embeddings: Batch of embeddings
        
    Returns:
        np.ndarray with shape (batch_size, dim)
    """
    # Convert to numpy
    try:
        import torch
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        elif isinstance(embeddings, list):
            pass  # Handle below
    except ImportError:
        if isinstance(embeddings, list):
            pass  # Handle below
    
    if isinstance(embeddings, list):
        # Handle list of embeddings
        if len(embeddings) == 0:
            return np.array([])
        
        # Normalize each embedding first
        normalized = [normalize_embedding_shape(emb) for emb in embeddings]
        return np.array(normalized)
    
    # Ensure numpy array
    if not isinstance(embeddings, np.ndarray):
        raise ValueError(f"Cannot normalize batch of type {type(embeddings)}")
    
    # Handle different shapes
    if embeddings.ndim == 1:
        # Single embedding, add batch dimension
        return embeddings.reshape(1, -1)
    elif embeddings.ndim == 2:
        # Already correct shape
        return embeddings
    elif embeddings.ndim == 3 and embeddings.shape[1] == 1:
        # Shape (batch, 1, dim) -> (batch, dim)
        return embeddings.squeeze(1)
    else:
        raise ValueError(f"Cannot normalize batch with shape {embeddings.shape}")


def validate_embedding_dimension(embedding: np.ndarray, expected_dim: int = 384) -> bool:
    """
    Validate that embedding has the expected dimension.
    
    Args:
        embedding: Embedding array
        expected_dim: Expected dimension (default: 384)
        
    Returns:
        True if dimension matches
    """
    if embedding.ndim == 1:
        return embedding.shape[0] == expected_dim
    elif embedding.ndim == 2:
        return embedding.shape[1] == expected_dim
    return False

if _EMB_UTIL_DIAG:
    print('[embedding_utils] module import end', flush=True)