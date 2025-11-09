"""Lightweight embedder with graceful fallbacks (no network by default)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:  # optional
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

try:  # optional
    import torch
    from transformers import AutoModel, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    AutoModel = AutoTokenizer = None  # type: ignore


@dataclass
class EmbeddingResult:
    vectors: np.ndarray
    model_name: Optional[str]
    normalized: bool


class Embedder:
    def __init__(self, model_name: Optional[str], normalize: bool = True, cache_dir: Optional[str] = None) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self.cache_dir = cache_dir
        self.model = None
        self.hf_model = None
        self.hf_tokenizer = None
        self.device = None

        if model_name:
            if model_name.startswith("hf:") and AutoModel and AutoTokenizer and torch is not None:
                base = model_name.split(":", 1)[1]
                try:
                    self.hf_tokenizer = AutoTokenizer.from_pretrained(base, cache_dir=cache_dir)
                    self.hf_model = AutoModel.from_pretrained(base, cache_dir=cache_dir)
                    self.hf_model.eval()
                    self.device = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu")
                    self.hf_model.to(self.device)
                except Exception as exc:  # pragma: no cover
                    logger.warning("HF model init failed (%s). Falling back to random.", exc)
                    self.hf_model = None
                    self.hf_tokenizer = None
            elif SentenceTransformer is not None:
                try:
                    self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
                except Exception as exc:  # pragma: no cover
                    logger.warning("SentenceTransformer init failed (%s). Falling back to random.", exc)
                    self.model = None

    def encode(self, texts: Iterable[str]) -> EmbeddingResult:
        items: List[str] = list(texts)
        if not items:
            return EmbeddingResult(np.zeros((0, 0), dtype=np.float32), self.model_name, self.normalize)

        vecs: np.ndarray
        if self.hf_model is not None and self.hf_tokenizer is not None and torch is not None:  # pragma: no cover
            with torch.no_grad():
                batch = self.hf_tokenizer(items, padding=True, truncation=True, max_length=256, return_tensors="pt")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.hf_model(**batch)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    hidden = outputs.pooler_output
                else:
                    hidden = outputs.last_hidden_state[:, 0, :]
                vecs = hidden.cpu().numpy().astype(np.float32)
        elif self.model is not None:  # pragma: no cover
            vecs = np.asarray(self.model.encode(items, show_progress_bar=False), dtype=np.float32)
        else:
            # Text-dependent deterministic random embeddings (no network)
            import hashlib

            dim = 384
            out: List[np.ndarray] = []
            for text in items:
                h = hashlib.sha256(text.encode("utf-8")).digest()
                seed = int.from_bytes(h[:8], byteorder="little", signed=False)
                rng = np.random.default_rng(seed)
                out.append(rng.normal(size=dim).astype(np.float32))
            vecs = np.stack(out, axis=0)

        if self.normalize and vecs.size > 0:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            vecs = vecs / norms
        return EmbeddingResult(vectors=vecs, model_name=self.model_name, normalized=self.normalize)
