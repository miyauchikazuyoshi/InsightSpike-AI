"""
Layer 2 Compatibility Module
===========================

Provides backward compatibility for old Layer 2 memory manager APIs.

DIAG: Adds lightweight import start/end markers when INSIGHTSPIKE_DIAG_IMPORT=1
to localize potential import-time hangs.
"""

import os
if os.getenv('INSIGHTSPIKE_DIAG_IMPORT') == '1':
    print('[layer2_compatibility] module import start', flush=True)

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
        # Direct Episode instance (newer pathway from MainAgent.add_knowledge)
        try:
            from insightspike.core.episode import Episode  # local import to avoid cycles
            if 'Episode' in globals() and isinstance(vec, Episode):  # already imported
                ep = vec
            elif isinstance(vec, Episode):
                ep = vec
            else:
                ep = None
        except Exception:
            ep = None
        if ep is not None:
            # Append directly if not already stored
            self.episodes.append(ep)
            idx = len(self.episodes) - 1
            # Ensure embedding dtype/shape
            if isinstance(ep.vec, np.ndarray) and ep.vec.dtype != np.float32:
                ep.vec = ep.vec.astype(np.float32)
            # Update vector index
            try:
                self._update_index(ep, idx)
            except Exception:
                pass
            return idx

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

    # Compatibility alias: some tests call l2_memory.retrieve(query_text, top_k)
    def retrieve(self, query_text: str, top_k: int = 5):
        try:
            res = self.search_episodes(query_text, k=top_k)
            return res or []
        except Exception:
            return []
    
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
        # Normalize shape (1, D) -> (D,)
        if isinstance(embedding, np.ndarray) and embedding.ndim == 2 and embedding.shape[0] == 1:
            embedding = embedding.reshape(-1)
        return embedding


# Convenience function for tests
def create_compatible_memory(dim: int = 384, config=None) -> CompatibleL2MemoryManager:
    """Create backward-compatible memory manager"""
    return CompatibleL2MemoryManager(dim, config)

if os.getenv('INSIGHTSPIKE_DIAG_IMPORT') == '1':
    print('[layer2_compatibility] module import end', flush=True)
