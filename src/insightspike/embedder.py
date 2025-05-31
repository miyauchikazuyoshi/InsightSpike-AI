"""Sentence‑Transformer embedder"""
import os
import warnings
from sentence_transformers import SentenceTransformer

# Import from the legacy config for compatibility
try:
    from .config import EMBED_MODEL_NAME
except ImportError:
    # Fallback if new config structure doesn't have it
    EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

__all__ = ["get_model"]

_model = None

def get_model():
    global _model
    if _model is None:
        model_name = os.getenv("EMBED_MODEL_PATH", EMBED_MODEL_NAME)
        try:
            _model = SentenceTransformer(model_name)
            print("📦 Embedding model loaded")
        except OSError as e:
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}'. "
                "Ensure the model files are available offline or set EMBED_MODEL_PATH."
            ) from e
        except Exception as e:
            # GPU失敗時はCPUにフォールバック
            warnings.warn(f"GPU loading failed, falling back to CPU: {e}")
            _model = SentenceTransformer(model_name, device="cpu")
            print("📦 Embedding model loaded on CPU (fallback)")
    return _model
