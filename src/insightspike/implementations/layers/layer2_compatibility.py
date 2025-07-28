"""
Layer 2 Compatibility Module
===========================

Provides backward compatibility for old Layer 2 memory manager APIs.
"""

from typing import Any, Dict, Optional, Union

import numpy as np

from .layer2_memory_manager import L2MemoryManager, MemoryConfig, MemoryMode


class CompatibleL2MemoryManager(L2MemoryManager):
    """
    Backward-compatible Layer 2 Memory Manager.

    Supports old API signatures while using new implementation.
    """

    def __init__(self, dim: int = 384, config=None):
        """Initialize with old-style parameters"""
        # Create MemoryConfig from old parameters
        memory_config = MemoryConfig.from_mode(MemoryMode.SCALABLE)
        memory_config.embedding_dim = dim

        # Initialize parent with new config
        super().__init__(memory_config)
        
        # Initialize embedder attribute
        self.embedder = None

    def add_episode(
        self,
        vec: Union[np.ndarray, str],
        text: Optional[str] = None,
        c_value: float = 0.5,
        metadata: Optional[Dict] = None,
    ) -> int:
        """
        Add episode with old API signature.

        Handles both:
        - add_episode(vec, text) - old style
        - add_episode(text, metadata) - new style
        """
        # Handle new-style call (first arg is text)
        if isinstance(vec, str) and text is None:
            return self.store_episode(vec, c_value, metadata)

        # Handle old-style call (first arg is vector)
        if isinstance(vec, np.ndarray) and text is not None:
            # Store the episode with provided vector
            # Note: This will replace the vector later, but maintains compatibility
            idx = self.store_episode(text, c_value, metadata)

            # Replace the auto-generated embedding with provided one
            if 0 <= idx < len(self.episodes):
                self.episodes[idx].vec = vec.astype(np.float32)
                # Rebuild index to include new vector
                self._rebuild_index()

            return idx

        # Fallback
        return -1
    
    def _encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to embedding vector.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector as numpy array with shape (embedding_dim,)
        """
        # Use the embedder from parent class
        if self.embedder is None:
            from insightspike.processing.embedder import EmbeddingManager
            self.embedder = EmbeddingManager()
        
        embedding = self.embedder.encode(text)
        # Shape is now normalized by EmbeddingManager
        return embedding


# Convenience function for tests
def create_compatible_memory(dim: int = 384, config=None) -> CompatibleL2MemoryManager:
    """Create backward-compatible memory manager"""
    return CompatibleL2MemoryManager(dim, config)
