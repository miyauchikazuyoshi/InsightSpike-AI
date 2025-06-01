"""
L2 Memory Manager - Compatibility Layer
=====================================

This module provides backward compatibility for the old Memory class.
New code should use: from insightspike.core.layers.layer2_memory_manager import L2MemoryManager

DEPRECATED: This file will be removed in a future version.
Use the new structured approach in core/layers/ instead.
"""

from __future__ import annotations
import warnings
from pathlib import Path
from typing import List
import json
import os

# Issue deprecation warning
warnings.warn(
    "insightspike.layer2_memory_manager is deprecated. "
    "Use insightspike.core.layers.layer2_memory_manager instead.",
    DeprecationWarning,
    stacklevel=2
)

try:
    import faiss
except ImportError:
    warnings.warn("FAISS not available, using fallback memory")
    faiss = None

import numpy as np

# Import the new implementations
from .core.layers.layer2_memory_manager import L2MemoryManager, Episode as NewEpisode

# Compatibility aliases
Episode = NewEpisode

from .embedder import get_model

# Import from the legacy config.py file using the same pattern as other modules
import importlib.util
try:
    # Import from the legacy config.py file explicitly 
    _config_file = os.path.join(os.path.dirname(__file__), 'config.py')
    _spec = importlib.util.spec_from_file_location("legacy_config", _config_file)
    _config = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_config)
    INDEX_FILE = _config.INDEX_FILE
    LAYER2_TOP_K = getattr(_config, 'LAYER2_TOP_K', 15)  # Import new Layer2 topK setting
except (ImportError, AttributeError):
    # Fallback for testing or if config is not available
    from pathlib import Path
    INDEX_FILE = Path("data/index.faiss")
    LAYER2_TOP_K = 15  # Fallback to optimized value

__all__ = ["Episode", "Memory"]


class Memory:
    """
    Compatibility wrapper for the old Memory class.
    
    This wraps the new L2MemoryManager to provide backward compatibility.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._manager = L2MemoryManager(dim)
        self.episodes: List[Episode] = []

    # ── construction ────────────────────────────────
    @classmethod
    def build(cls, docs: List[str], batch_size=32):
        """Build memory from documents using the new L2MemoryManager."""
        manager = L2MemoryManager.build_from_documents(docs, batch_size)
        memory = cls(manager.dim)
        memory._manager = manager
        return memory

    @classmethod  
    def load(cls):
        """Load memory from file using the new L2MemoryManager."""
        try:
            manager = L2MemoryManager.load_from_file(INDEX_FILE)
            memory = cls(manager.dim)
            memory._manager = manager
            return memory
        except FileNotFoundError:
            # Fallback to empty memory
            return cls(384)  # Default dimension

    # ── retrieval ────────────────────────────────
    def topk(self, q_vec: np.ndarray, k: int = None) -> List[Episode]:
        """Top-k retrieval using the new L2MemoryManager."""
        if k is None:
            k = LAYER2_TOP_K
            
        results = self._manager.retrieve(q_vec, k)
        return [Episode(vec=r.vec, text=r.text, c=r.c) for r in results]

    def save(self, path: Path = None):
        """Save memory using the new L2MemoryManager."""
        save_path = path or INDEX_FILE
        self._manager.save_to_file(save_path)

    def train_index(self):
        """Train the index - handled internally by L2MemoryManager."""
        pass  # New implementation handles this automatically


# Export compatibility symbols
__all__ = ["Episode", "Memory"]
