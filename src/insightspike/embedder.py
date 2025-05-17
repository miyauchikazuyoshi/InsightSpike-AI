"""Sentence‑Transformer embedder"""
from sentence_transformers import SentenceTransformer
from .config import EMBED_MODEL_NAME

__all__ = ["get_model"]

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    return _model