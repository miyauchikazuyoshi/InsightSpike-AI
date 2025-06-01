"""Sentence‑Transformer embedder"""
import os
import warnings
import numpy as np

# Lite mode check
if os.getenv('INSIGHTSPIKE_LITE_MODE') == '1':
    # Lite mode: use mock SentenceTransformer
    class SentenceTransformer:
        def __init__(self, model_name=None):
            self.model_name = model_name or 'mock-model'
        
        def encode(self, sentences, **kwargs):
            # Return dummy embeddings
            if isinstance(sentences, str):
                sentences = [sentences]
            return np.random.rand(len(sentences), 384).astype('float32')
else:
    # Full mode: use real SentenceTransformer
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        # Fallback to mock if not available
        class SentenceTransformer:
            def __init__(self, model_name=None):
                self.model_name = model_name or 'fallback-model'
            
            def encode(self, sentences, **kwargs):
                if isinstance(sentences, str):
                    sentences = [sentences]
                return np.random.rand(len(sentences), 384).astype('float32')

# Import from the new config for compatibility
try:
    from .core.config import get_config
    config = get_config()
    EMBED_MODEL_NAME = config.embedding.model_name
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

class Embedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts)
    
    def embed_single(self, text):
        return self.embed([text])[0]
