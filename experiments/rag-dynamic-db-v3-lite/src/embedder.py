"""Sentence embedding utilities for RAG v3-lite."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import torch
    from transformers import AutoModel, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    AutoModel = AutoTokenizer = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    vectors: np.ndarray
    model_name: Optional[str]
    normalized: bool


class Embedder:
    """Wrapper around SentenceTransformer with graceful fallback."""

    def __init__(self, model_name: Optional[str], normalize: bool = True, cache_dir: Optional[str] = None) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self.cache_dir = cache_dir
        self.model = None
        self.hf_model = None
        self.hf_tokenizer = None
        self.device = None

        if model_name:
            if model_name.startswith("hf:"):
                base_model = model_name.split(":", 1)[1]
                if AutoModel is None or AutoTokenizer is None or torch is None:
                    logger.warning(
                        "transformers が利用できないため、`%s` を初期化できません。ランダム埋め込みにフォールバックします。",
                        base_model,
                    )
                else:
                    try:
                        self.hf_tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir)
                        self.hf_model = AutoModel.from_pretrained(base_model, cache_dir=cache_dir)
                        self.hf_model.eval()
                        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        self.hf_model.to(self.device)
                    except Exception as exc:  # pragma: no cover
                        logger.warning(
                            "HuggingFaceモデル `%s` の初期化に失敗しました (%s)。ランダム埋め込みを使用します。",
                            base_model,
                            exc,
                        )
                        self.hf_model = None
                        self.hf_tokenizer = None
            else:
                if SentenceTransformer is None:
                    logger.warning(
                        "sentence-transformers が見つからないため、ランダム埋め込みにフォールバックします。 "
                        "本番環境では `pip install sentence-transformers` を実行してください。"
                    )
                else:
                    try:
                        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
                    except Exception as exc:  # pragma: no cover
                        logger.warning("SentenceTransformer の初期化に失敗しました (%s)。ランダム埋め込みを使用します。", exc)
                        self.model = None

    def encode(self, texts: Iterable[str]) -> EmbeddingResult:
        texts_list: List[str] = list(texts)
        if not texts_list:
            return EmbeddingResult(np.zeros((0, 0), dtype=np.float32), self.model_name, self.normalize)

        if self.hf_model is not None and self.hf_tokenizer is not None:
            with torch.no_grad():
                batch = self.hf_tokenizer(
                    texts_list,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.hf_model(**batch)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    hidden = outputs.pooler_output
                else:
                    hidden = outputs.last_hidden_state[:, 0, :]
                vectors = hidden.cpu().numpy().astype(np.float32)
        elif self.model is None:
            rng = np.random.default_rng(1234)
            vectors = rng.normal(size=(len(texts_list), 384)).astype(np.float32)
        else:
            vectors = np.asarray(self.model.encode(texts_list, show_progress_bar=False), dtype=np.float32)

        if self.normalize and vectors.size > 0:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
            vectors = vectors / norms

        return EmbeddingResult(vectors=vectors, model_name=self.model_name, normalized=self.normalize)
