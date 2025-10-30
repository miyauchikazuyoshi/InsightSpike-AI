from __future__ import annotations

"""Utilities to provide real embedder/LLM generator with graceful fallbacks.

Exposes two builder functions:

- build_embedder(kb_texts, queries, mode="auto") -> object with .encode(list[str])->np.ndarray
- build_generator(provider: str|None=None, model: str|None=None) -> object with .generate(query, context)->str

The embedder tries: sentence-transformers -> InsightSpike EmbeddingManager -> TF‑IDF -> random fallback.
The generator tries: InsightSpike ProviderFactory for (anthropic/openai) -> mock template.
"""

import os
from typing import Callable, List, Optional
from pathlib import Path

import numpy as np


def l2norm(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x / (np.linalg.norm(x) + 1e-9)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def build_embedder(kb_texts: List[str], queries: List[str], mode: str = "auto"):
    mode = (mode or "auto").lower()

    if mode == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer  # lazy import
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
        vec.fit(kb_texts + queries)

        class _TFIDF:
            def encode(self, texts: List[str]) -> np.ndarray:
                arr = vec.transform(texts).astype(np.float64).toarray()
                return l2norm(arr)

        return _TFIDF()

    # auto: try ST
    if mode in ("auto", "st", "sentence-transformers"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            model_name = os.getenv("GE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            st_model = SentenceTransformer(model_name)

            class _ST:
                def encode(self, texts: List[str]) -> np.ndarray:
                    return l2norm(np.asarray(st_model.encode(texts)))

            return _ST()
        except Exception:
            if mode == "st":
                # fallthrough to other modes
                pass

    # try InsightSpike EmbeddingManager
    if mode in ("auto", "insightspike", "is"):
        try:
            from insightspike.processing.embedder import EmbeddingManager  # type: ignore

            mgr = EmbeddingManager()

            class _IS:
                def encode(self, texts: List[str]) -> np.ndarray:
                    return l2norm(np.asarray(mgr.encode(texts, normalize_embeddings=True)))

            return _IS()
        except Exception:
            pass

    # fallback: tfidf
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
        vec.fit(kb_texts + queries)

        class _TFIDFFallback:
            def encode(self, texts: List[str]) -> np.ndarray:
                arr = vec.transform(texts).astype(np.float64).toarray()
                return l2norm(arr)

        return _TFIDFFallback()
    except Exception:
        pass

    # last resort: deterministic random
    class _Rand:
        def __init__(self, dim: int = 384):
            self.dim = dim

        def encode(self, texts: List[str]) -> np.ndarray:
            vecs = []
            for t in texts:
                seed = abs(hash(t)) % (2**32)
                rng = np.random.default_rng(seed)
                v = rng.normal(size=(self.dim,))
                v = v / (np.linalg.norm(v) + 1e-9)
                vecs.append(v)
            return np.vstack(vecs)

    return _Rand()


def choose_provider(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit.lower()
    env = os.environ
    if env.get("ANTHROPIC_API_KEY") or env.get("CLAUDE_API_KEY") or env.get("ANTH_API_KEY"):
        return "anthropic"
    if env.get("OPENAI_API_KEY"):
        return "openai"
    return "mock"


def build_generator(provider_name: Optional[str] = None, model_name: Optional[str] = None):
    name = choose_provider(provider_name)

    if name in ("anthropic", "openai"):
        try:
            from insightspike.providers.provider_factory import ProviderFactory  # type: ignore

            api_key = None
            if name == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY") or os.getenv("ANTH_API_KEY")
            elif name == "openai":
                api_key = os.getenv("OPENAI_API_KEY")

            provider = ProviderFactory.create(name, api_key=api_key)

            class _LLM:
                def generate(self, query: str, context: str) -> str:
                    prompt = (
                        "Answer the question using the following snippets.\n\n"
                        f"Context:\n{context}\n\nQuestion: {query}\n"
                    )
                    if model_name:
                        return provider.generate(prompt, model=model_name)
                    return provider.generate(prompt)

            return _LLM()
        except Exception:
            # fall back to mock
            pass

    # mock template generator
    class _Mock:
        def generate(self, query: str, context: str) -> str:
            snippets = [s.strip() for s in context.split("\n") if s.strip()]
            head = snippets[:3]
            if not head:
                return f"I don't have enough information to answer '{query}'."
            lines = ["Based on the available information:"] + [f"- {h}" for h in head]
            return "\n".join(lines)

    return _Mock()

