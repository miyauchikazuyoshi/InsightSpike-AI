"""
Embedder Utility - Enhanced Text Embedding Support
================================================

Provides flexible text embedding with multiple model support.

DIAG: When INSIGHTSPIKE_DIAG_IMPORT=1, prints start/end markers to help isolate
import-time stalls (particularly when layer2 import appears to hang).
"""

import os as _os
_EMB_DIAG = _os.getenv('INSIGHTSPIKE_DIAG_IMPORT') == '1'
if _EMB_DIAG:
    print('[embedder] module import start', flush=True)

import logging
import os
from typing import List, Optional, Union
# Potential numpy import stall mitigation and diagnostics
if _EMB_DIAG:
    print('[embedder] before numpy import', flush=True)
    # Limit BLAS / threading to avoid potential initialization hangs
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
try:
    import numpy as np
    if _EMB_DIAG:
        print('[embedder] after numpy import', flush=True)
except Exception as e:  # pragma: no cover
    if _EMB_DIAG:
        print(f'[embedder] numpy import failed: {e}', flush=True)
    raise

# Potential stall suspected at embedding_utils import; add pre-import marker and lite-mode skip
if _EMB_DIAG:
    print('[embedder] before embedding_utils import', flush=True)

if os.getenv('INSIGHTSPIKE_LITE_MODE') == '1' or os.getenv('INSIGHTSPIKE_MIN_IMPORT') == '1':
    # Define lightweight no-op normalizers to bypass potential import stall
    if _EMB_DIAG:
        print('[embedder] skipping embedding_utils (lite/min mode)', flush=True)

    def normalize_embedding_shape(x):  # type: ignore
        return x

    def normalize_batch_embeddings(x):  # type: ignore
        return x
else:
    from ..utils.embedding_utils import normalize_embedding_shape, normalize_batch_embeddings
    if _EMB_DIAG:
        print('[embedder] imported embedding_utils', flush=True)

logger = logging.getLogger(__name__)

__all__ = ["get_model", "EmbeddingManager"]

# Global model cache
_model_cache = {}
_global_manager = None

# Final module import completion marker for diagnostics
if _EMB_DIAG:
    try:
        print('[embedder] module import end', flush=True)
    except Exception:
        pass


class EmbeddingManager:
    """Manages text embedding models with caching and fallback support."""

    def __init__(self, model_name: str = None, config=None, weight_manager=None):
        from ..config.models import InsightSpikeConfig

        # Handle both Pydantic config and legacy config
        if config and isinstance(config, InsightSpikeConfig):
            self.config = config
            self.model_name = model_name or self.config.embedding.model_name
            self.dimension = self.config.embedding.dimension
        elif config:
            # Legacy config support
            try:
                self.config = config
                self.model_name = model_name or self.config.embedding.model_name
                self.dimension = self.config.embedding.dimension
            except AttributeError:
                # Fallback for incomplete config
                self.config = config
                self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
                self.dimension = 384
        else:
            # No config provided, use defaults
            self.config = None
            self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
            self.dimension = 384
        self._model = None
        self.weight_manager = weight_manager

    def get_model(self):
        """Get or initialize the embedding model."""
        if self._model is not None:
            return self._model

        global _model_cache

        if self.model_name in _model_cache:
            self._model = _model_cache[self.model_name]
            return self._model

        # --- Lightweight / Safe mode gates ---------------------------------
        # Global safe mode flag
        if os.getenv("INSIGHTSPIKE_SAFE_MODE") == "1":
            logger.info("Safe mode enabled, using fallback embedder")
            return self._fallback_model()

        # Explicit lite/minimal import flags (set in tests & quick CLI tools)
        if os.getenv("INSIGHTSPIKE_LITE_MODE") == "1" or os.getenv("INSIGHTSPIKE_MIN_IMPORT") == "1":
            if os.getenv("INSIGHTSPIKE_ENABLE_FULL_EMBEDDING") != "1":  # allow explicit override
                logger.debug("Lite/min mode detected -> using fallback embedder (skip model load)")
                return self._fallback_model()

        # PyTest heuristic (collection phase does not set PYTEST_CURRENT_TEST yet, so rely on lite flags above)
        if os.getenv("PYTEST_CURRENT_TEST") and os.getenv("INSIGHTSPIKE_ENABLE_FULL_EMBEDDING") != "1":
            logger.info("PyTest detected without full embedding override; using fallback embedder")
            return self._fallback_model()

        try:
            # Try sentence-transformers first with safe initialization
            import torch
            from sentence_transformers import SentenceTransformer

            # Optimize environment for CPU-only stable execution
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            torch.set_num_threads(1)

            logger.info(f"Loading embedding model: {self.model_name}")

            # Safe initialization with explicit parameters
            model = SentenceTransformer(
                self.model_name,
                device="cpu",
                cache_folder=None,  # Avoid cache conflicts
                trust_remote_code=False,
            )

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
        """Encode texts to embeddings with shape normalization."""
        single_text = isinstance(texts, str)
        if single_text:
            texts = [texts]

        model = self.get_model()

        if hasattr(model, "encode"):
            # Standard sentence-transformers interface
            # Force disable progress bar to avoid multiprocessing issues
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,  # Always disable to prevent issues
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize_embeddings,
            )
        else:
            # Fallback encoding
            embeddings = self._fallback_encode(texts)
        
        # Apply weight manager if available (for backward compatibility)
        if self.weight_manager:
            try:
                if isinstance(embeddings, np.ndarray):
                    embeddings = self.weight_manager.apply_to_batch(embeddings)
            except Exception as e:
                logger.warning(f"Weight application failed: {e}, using original embeddings")
        
        # Normalize shape
        if single_text:
            # Return shape (dim,) for single text
            return normalize_embedding_shape(embeddings[0] if len(embeddings) > 0 else embeddings)
        else:
            # Return shape (batch_size, dim) for multiple texts
            return normalize_batch_embeddings(embeddings)

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text with shape (dim,)."""
        return self.encode(text)  # encode now handles shape normalization

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

    def get_sentence_embedding_dimension(self):
        """Return the dimension of the embeddings."""
        return self.dimension

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

    try:
        # Ultra-lightweight path for tests / diagnostics: skip heavy imports entirely
        if os.getenv("INSIGHTSPIKE_MIN_IMPORT") == "1":  # pragma: no cover (simple guard)
            return FallbackEmbedder(getattr(_global_manager, 'dimension', 384))
        return _global_manager.get_model()
    except Exception as e:  # pragma: no cover
        logger.warning(f"Embedding get_model fallback due to: {e}")
        # Return lightweight fallback embedder instance
        return FallbackEmbedder(getattr(_global_manager, 'dimension', 384))

# Tail-end marker to confirm full module execution reached
if _EMB_DIAG:
    try:
        print('[embedder] module import tail end', flush=True)
    except Exception:
        pass
