"""Sentence‑Transformer embedder"""
import os
from sentence_transformers import SentenceTransformer
from .config import EMBED_MODEL_NAME

__all__ = ["get_model"]

_model = None

def get_model():
    global _model
    if _model is None:
        model_name = os.getenv("EMBED_MODEL_PATH", EMBED_MODEL_NAME)
        try:
            _model = SentenceTransformer(model_name, device="cpu")
        except OSError as e:
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}'. "
                "Ensure the model files are available offline or set EMBED_MODEL_PATH."
            ) from e
    return _model
