"""
Embedder Interface
==================

Protocol for text embedding implementations.
"""

from typing import Protocol, List, Union, runtime_checkable
import numpy as np


@runtime_checkable
class IEmbedder(Protocol):
    """
    Interface for text embedders.
    
    All embedders must return normalized shapes:
    - Single text: shape (dimension,)
    - Multiple texts: shape (batch_size, dimension)
    """
    
    @property
    def dimension(self) -> int:
        """
        Get embedding dimension.
        
        Returns:
            Embedding dimension
        """
        ...
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encode text(s) to embeddings.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress
            
        Returns:
            Embeddings with shape (dim,) or (batch_size, dim)
        """
        ...
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding with shape (dimension,)
        """
        ...