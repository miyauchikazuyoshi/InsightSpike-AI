"""Hybrid BM25 + embedding retriever (self-contained)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .embedder import Embedder, EmbeddingResult


@dataclass
class Document:
    doc_id: str
    text: str
    embedding: np.ndarray
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class RetrievalHit:
    document: Document
    score: float
    bm25_score: float
    embedding_score: float
    rank: int


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        bm25_weight: float = 0.5,
        embedding_weight: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.embedder = embedder
        self.bm25_weight = bm25_weight
        self.embedding_weight = embedding_weight
        self.rng = np.random.default_rng(seed)
        self.last_query_embedding: np.ndarray | None = None

        self.documents: Dict[str, Document] = {}
        self.doc_embeddings: np.ndarray | None = None
        self.term_freqs: Dict[str, Dict[str, int]] = {}
        self.doc_freq: Dict[str, int] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0

        self.k1 = 1.2
        self.b = 0.75

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import re

        return re.findall(r"\w+", text.lower())

    def add_corpus(self, docs: Iterable[Tuple[str, str, Dict[str, str]]]) -> None:
        doc_list = list(docs)
        vectors = self.embedder.encode([text for _, text, _ in doc_list]).vectors
        for idx, (doc_id, text, metadata) in enumerate(doc_list):
            embedding = vectors[idx] if idx < len(vectors) else self.rng.normal(size=384).astype(np.float32)
            tokens = self._tokenize(text)
            self.documents[doc_id] = Document(doc_id=doc_id, text=text, embedding=embedding, metadata=metadata or {})
            self.doc_lengths[doc_id] = len(tokens)
            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.term_freqs[doc_id] = tf
            for token in set(tokens):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
        if self.doc_lengths:
            self.avg_doc_length = float(np.mean(list(self.doc_lengths.values())))
        self.doc_embeddings = np.stack([doc.embedding for doc in self.documents.values()], axis=0)

    def _bm25(self, query_tokens: List[str], doc_id: str) -> float:
        tf = self.term_freqs.get(doc_id)
        if tf is None:
            return 0.0
        score = 0.0
        doc_length = self.doc_lengths.get(doc_id, 1)
        N = max(1, len(self.documents))
        for token in query_tokens:
            if token not in tf:
                continue
            freq = tf[token]
            df = self.doc_freq.get(token, 0)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1e-9))
            score += idf * (numerator / denominator)
        return float(score)

    def _embedding_scores(self, query_vector: np.ndarray) -> np.ndarray:
        if self.doc_embeddings is None or self.doc_embeddings.size == 0:
            return np.zeros((0,), dtype=np.float32)
        scores = np.dot(self.doc_embeddings, query_vector)
        return scores.astype(np.float32)

    def get_last_query_embedding(self) -> np.ndarray | None:
        if self.last_query_embedding is None:
            return None
        return np.asarray(self.last_query_embedding, dtype=np.float32).copy()

    def retrieve(self, query: str, top_k: int = 4) -> List[RetrievalHit]:
        query_tokens = self._tokenize(query)
        query_vec = self.embedder.encode([query]).vectors[0]
        self.last_query_embedding = np.asarray(query_vec, dtype=np.float32).copy()

        bm25_scores = {doc_id: self._bm25(query_tokens, doc_id) for doc_id in self.documents}
        embed_scores = self._embedding_scores(query_vec)

        combined: List[Tuple[float, float, float, Document]] = []
        doc_ids = list(self.documents.keys())
        for idx, doc_id in enumerate(doc_ids):
            doc = self.documents[doc_id]
            emb_score = float(embed_scores[idx]) if idx < len(embed_scores) else 0.0
            bm_score = float(bm25_scores.get(doc_id, 0.0))
            score = self.bm25_weight * bm_score + self.embedding_weight * emb_score
            combined.append((score, bm_score, emb_score, doc))

        combined.sort(key=lambda x: x[0], reverse=True)
        hits: List[RetrievalHit] = []
        for rank, (score, bm_score, emb_score, doc) in enumerate(combined[:top_k], start=1):
            hits.append(
                RetrievalHit(
                    document=doc,
                    score=float(score),
                    bm25_score=float(bm_score),
                    embedding_score=float(emb_score),
                    rank=rank,
                )
            )
        return hits

