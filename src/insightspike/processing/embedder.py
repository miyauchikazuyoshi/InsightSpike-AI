"""
Embedder Utility - Enhanced Text Embedding Support
================================================

Provides flexible text embedding with multiple model support.
"""

import logging
import os
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["get_model", "EmbeddingManager"]

# Global model cache
_model_cache = {}
_global_manager = None


class EmbeddingManager:
    """Manages text embedding models with caching and fallback support."""

    def __init__(self, model_name: str = None, config=None):
        try:
            from ...config import get_config

            self.config = config or get_config()
            self.model_name = model_name or self.config.embedding.model_name
            self.dimension = self.config.embedding.dimension
        except ImportError:
            # Fallback configuration
            self.config = None
            self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
            self.dimension = 384
        self._model = None

    def get_model(self):
        """Get or initialize the embedding model."""
        if self._model is not None:
            return self._model

        global _model_cache

        if self.model_name in _model_cache:
            self._model = _model_cache[self.model_name]
            return self._model

        # Check for safe mode
        if os.getenv("INSIGHTSPIKE_SAFE_MODE") == "1":
            logger.info("Safe mode enabled, using fallback embedder")
            return self._fallback_model()

        try:
            # Try sentence-transformers first
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            # Try to create model, handle device parameter issues gracefully
            try:
                # Explicitly use CPU to avoid GPU segfaults
                model = SentenceTransformer(self.model_name, device="cpu")
            except (TypeError, ValueError) as device_error:
                # If device parameter is not supported, try without it
                logger.warning(
                    f"Device parameter not supported: {device_error}. Trying without device parameter."
                )
                model = SentenceTransformer(self.model_name)

            _model_cache[self.model_name] = model
            self._model = model

            return model

        except ImportError:
            logger.error(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
            return self._fallback_model()
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            return self._fallback_model()

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Encode texts to embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        model = self.get_model()

        if hasattr(model, "encode"):
            # Standard sentence-transformers interface
            return model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize_embeddings,
            )
        else:
            # Fallback encoding
            return self._fallback_encode(texts)

    def _fallback_model(self):
        """Fallback to simple hash-based model."""
        logger.warning("Using fallback hash-based embedding model")

        global _model_cache
        fallback_key = f"fallback_{self.dimension}"

        if fallback_key not in _model_cache:
            _model_cache[fallback_key] = FallbackEmbedder(self.dimension)

        self._model = _model_cache[fallback_key]
        return _model_cache[fallback_key]

    def _fallback_encode(self, texts: List[str]) -> np.ndarray:
        """Fallback encoding using simple methods."""
        embeddings = []

        for text in texts:
            # Simple hash-based embedding
            hash_val = hash(text) % (2**32)
            # Convert to normalized vector
            np.random.seed(hash_val)
            embedding = np.random.normal(0, 1, self.dimension)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32)


class FallbackEmbedder:
    """Simple fallback embedder for when sentence-transformers is unavailable."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Generate simple hash-based embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            # Use text hash as seed for reproducible embeddings
            hash_val = hash(text) % (2**32)
            np.random.seed(hash_val)

            # Generate normalized random vector
            embedding = np.random.normal(0, 1, self.dimension)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32)


# Global convenience function
def get_model(model_name: str = None):
    """Get global embedding model instance."""
    global _global_manager

    if _global_manager is None:
        _global_manager = EmbeddingManager(model_name)
    elif model_name and _global_manager.model_name != model_name:
        # If a different model is requested, create a new manager
        _global_manager = EmbeddingManager(model_name)

    return _global_manager.get_model()
