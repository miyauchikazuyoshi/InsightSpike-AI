"""Dense retriever using E5-base-v2 embeddings + FAISS IndexFlatIP.

Provides:
  - Offline index building (build_index)
  - Fast dense retrieval (retrieve)
  - Embedding lookup for graph Tier D edge construction (get_doc_embeddings)

Usage::

    from dense_retriever import DenseRetriever

    dr = DenseRetriever(index_dir="data/bright/dense_index")
    dr.load_index("biology")
    results = dr.retrieve("What causes phototaxis?", "biology", top_k=100)
    embeddings = dr.get_doc_embeddings("biology", ["doc_id_1", "doc_id_2"])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class DenseRetriever:
    """E5-base-v2 embedding + FAISS IndexFlatIP for dense retrieval.

    Parameters
    ----------
    model_name : str
        SentenceTransformer model name (default: intfloat/e5-base-v2).
    index_dir : str | Path | None
        Directory containing pre-built FAISS indices.
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        index_dir: str | Path | None = None,
        device: str = "mps",
    ):
        self.model_name = model_name
        self.index_dir = Path(index_dir) if index_dir else None
        self.device = device
        self._model = None  # lazy-loaded
        self.indices: dict = {}        # domain -> faiss.Index
        self.doc_ids: dict = {}        # domain -> list[str]
        self.embeddings: dict = {}     # domain -> np.ndarray
        self._id_to_idx: dict = {}     # domain -> dict[str, int]

    @property
    def model(self):
        """Lazy-load the SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SentenceTransformer: %s (device=%s)",
                        self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def build_index(self, docs_jsonl_path: str | Path, domain: str) -> None:
        """Encode all docs and build a FAISS IndexFlatIP index.

        Saves to ``index_dir/{domain}.faiss``, ``{domain}_ids.json``,
        and ``{domain}_emb.npy``.

        Parameters
        ----------
        docs_jsonl_path : str | Path
            Path to the JSONL file with ``{"id": ..., "content": ...}`` docs.
        domain : str
            Domain name (e.g. "biology").
        """
        # NOTE: faiss is imported AFTER encoding to avoid AVX2/torch conflicts
        docs_path = Path(docs_jsonl_path)
        logger.info("Building dense index for domain=%s from %s", domain, docs_path)

        # Load document IDs and raw content
        doc_ids_list: list[str] = []
        raw_lines: list[str] = []
        with open(docs_path) as f:
            for line in f:
                raw_lines.append(line)
                doc = json.loads(line)
                doc_ids_list.append(doc["id"])

        n_docs = len(doc_ids_list)
        logger.info("  %d documents loaded", n_docs)

        # Pre-init model before chunked encoding
        _ = self.model
        logger.info("  Encoding with %s (device=%s) ...", self.model_name, self.device)

        # Encode in chunks to avoid memory issues
        chunk_size = 1000
        batch_size = 16 if self.device == "mps" else 64
        dim = None
        all_embeddings: list[np.ndarray] = []

        for start in range(0, n_docs, chunk_size):
            end = min(start + chunk_size, n_docs)
            logger.info("    Encoding chunk %d-%d / %d ...", start, end, n_docs)
            chunk_texts = []
            for i in range(start, end):
                doc = json.loads(raw_lines[i])
                chunk_texts.append("passage: " + doc["content"][:480])
            chunk_emb = self.model.encode(
                chunk_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            all_embeddings.append(chunk_emb.astype(np.float32))
            if dim is None:
                dim = chunk_emb.shape[1]
            del chunk_texts, chunk_emb

        del raw_lines
        embeddings = np.vstack(all_embeddings)
        del all_embeddings
        logger.info("  Embeddings shape: %s", embeddings.shape)

        # Build FAISS index AFTER encoding (import here to avoid AVX2/torch conflicts)
        import faiss
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        logger.info("  FAISS index built: %d vectors, dim=%d", index.ntotal, dim)

        # Save
        if self.index_dir is None:
            raise ValueError("index_dir must be set to save index")
        self.index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(self.index_dir / f"{domain}.faiss"))
        with open(self.index_dir / f"{domain}_ids.json", "w") as f:
            json.dump(doc_ids_list, f)
        np.save(str(self.index_dir / f"{domain}_emb.npy"), embeddings)

        logger.info("  Saved to %s", self.index_dir)

        # Keep in memory
        self.indices[domain] = index
        self.doc_ids[domain] = doc_ids_list
        self.embeddings[domain] = embeddings
        self._id_to_idx[domain] = {did: i for i, did in enumerate(doc_ids_list)}

    def load_index(self, domain: str) -> None:
        """Load pre-built index + embeddings from disk.

        Parameters
        ----------
        domain : str
            Domain name (e.g. "biology").
        """
        import faiss

        if self.index_dir is None:
            raise ValueError("index_dir must be set to load index")

        idx_path = self.index_dir / f"{domain}.faiss"
        ids_path = self.index_dir / f"{domain}_ids.json"
        emb_path = self.index_dir / f"{domain}_emb.npy"

        logger.info("Loading dense index for domain=%s from %s", domain, self.index_dir)

        self.indices[domain] = faiss.read_index(str(idx_path))
        with open(ids_path) as f:
            self.doc_ids[domain] = json.load(f)
        self.embeddings[domain] = np.load(str(emb_path))
        self._id_to_idx[domain] = {
            did: i for i, did in enumerate(self.doc_ids[domain])
        }

        logger.info(
            "  Loaded: %d docs, dim=%d",
            len(self.doc_ids[domain]),
            self.embeddings[domain].shape[1],
        )

    def retrieve(
        self,
        query: str,
        domain: str,
        top_k: int = 100,
        exclude_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Dense retrieval for a query.

        Parameters
        ----------
        query : str
            Query text.
        domain : str
            Domain to search in.
        top_k : int
            Number of results to return.
        exclude_ids : set[str] | None
            Document IDs to exclude from results.

        Returns
        -------
        list[tuple[str, float]]
            (doc_id, similarity_score) pairs, highest first.
        """
        if domain not in self.indices:
            raise KeyError(f"Index not loaded for domain: {domain}")

        # E5 format: "query: {text}"
        q_emb = self.model.encode(
            ["query: " + query[:480]],
            normalize_embeddings=True,
        ).astype(np.float32)

        # Over-fetch to handle exclusions
        fetch_k = top_k + (len(exclude_ids) if exclude_ids else 0) + 50
        fetch_k = min(fetch_k, self.indices[domain].ntotal)

        scores, indices = self.indices[domain].search(q_emb, fetch_k)
        scores = scores[0]
        indices = indices[0]

        doc_ids_list = self.doc_ids[domain]
        exclude = exclude_ids or set()

        results: list[tuple[str, float]] = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            did = doc_ids_list[idx]
            if did in exclude:
                continue
            results.append((did, float(score)))
            if len(results) >= top_k:
                break

        return results

    def get_doc_embeddings(
        self, domain: str, doc_ids: list[str]
    ) -> dict[str, np.ndarray]:
        """Return pre-computed embeddings for specific document IDs.

        Used for constructing Tier D (Dense similarity) graph edges.

        Parameters
        ----------
        domain : str
            Domain name.
        doc_ids : list[str]
            Document IDs to look up.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from doc_id to its embedding vector (L2-normalized).
        """
        if domain not in self._id_to_idx:
            return {}

        id_to_idx = self._id_to_idx[domain]
        emb_matrix = self.embeddings[domain]
        result: dict[str, np.ndarray] = {}

        for did in doc_ids:
            idx = id_to_idx.get(did)
            if idx is not None:
                result[did] = emb_matrix[idx]

        return result
