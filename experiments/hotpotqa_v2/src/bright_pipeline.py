"""v12: BRIGHT Benchmark — Topology-Enhanced Retrieval Pipeline.

Graph-based re-ranking for reasoning-intensive retrieval:
  1. BM25 initial retrieval (top-100 from domain corpus)
  2. Entity graph construction from top-N documents
  3. Graph-based re-ranking: centrality + query-distance scoring
  4. Optional β₀-driven bridge expansion

Evaluation metric: nDCG@10

Usage::

    from bright_pipeline import BrightPipeline, build_bm25_index
    index, docs = build_bm25_index("data/bright/biology_docs.jsonl")
    pipeline = BrightPipeline()
    result = pipeline.rerank(query, index, docs, gold_ids=["doc1", "doc2"])
"""

from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx

from entity_graph import extract_entities, build_sentence_graph


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BrightResult:
    """Result of BRIGHT re-ranking for a single query."""

    query_id: str
    ranked_doc_ids: list[str]       # re-ranked document IDs (top-k)
    ranked_scores: list[float]      # re-ranking scores
    bm25_doc_ids: list[str]         # original BM25 ranking (for comparison)
    beta_0: int = 0
    beta_1: int = 0
    n_graph_nodes: int = 0
    n_graph_edges: int = 0
    n_docs_in_graph: int = 0
    latency_ms: float = 0.0
    entity_feval_applied: bool = False


# ---------------------------------------------------------------------------
# BM25 Tokenizer (shared between index and query)
# ---------------------------------------------------------------------------

import re as _re

# Lazy-loaded globals for BM25 tokenization
_bm25_stopwords: set[str] | None = None
_bm25_stemmer = None


def _get_bm25_stopwords() -> set[str]:
    """Lazy-load NLTK English stopwords."""
    global _bm25_stopwords
    if _bm25_stopwords is None:
        import nltk
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        _bm25_stopwords = set(stopwords.words("english"))
    return _bm25_stopwords


def _get_bm25_stemmer():
    """Lazy-load Porter stemmer."""
    global _bm25_stemmer
    if _bm25_stemmer is None:
        from nltk.stem import PorterStemmer
        _bm25_stemmer = PorterStemmer()
    return _bm25_stemmer


# When using Pyserini, Lucene handles tokenization internally.
# Set this to True to skip Python-side stemming/stopwords.
_bm25_use_pyserini: bool = False


def set_bm25_pyserini_mode(enabled: bool = True) -> None:
    """Switch bm25_tokenize behavior for Pyserini (Lucene-native tokenization)."""
    global _bm25_use_pyserini
    _bm25_use_pyserini = enabled


def bm25_tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 with stopword removal and stemming.

    When Pyserini mode is enabled, performs minimal tokenization
    (Lucene handles stemming and stopwords internally).

    Steps (rank_bm25 mode):
      1. Lowercase
      2. Extract alphanumeric tokens (handles punctuation, hyphens)
      3. Remove stopwords
      4. Remove single-char tokens
      5. Apply Porter stemming

    Steps (pyserini mode):
      1. Return words as-is (Lucene Analyzer handles the rest)
    """
    if _bm25_use_pyserini:
        # Minimal tokenization — Lucene's DefaultEnglishAnalyzer
        # handles lowercasing, stemming, and stopword removal
        return text.split()

    tokens = _re.findall(r"[a-z0-9]+", text.lower())
    sw = _get_bm25_stopwords()
    stemmer = _get_bm25_stemmer()
    return [stemmer.stem(t) for t in tokens if t not in sw and len(t) > 1]


# ---------------------------------------------------------------------------
# BM25 Index Builder (document-level)
# ---------------------------------------------------------------------------

def build_bm25_index(
    docs_path: str,
    engine: str = "rank_bm25",
    lucene_index_path: str | None = None,
    pyserini_k1: float = 0.9,
    pyserini_b: float = 0.4,
) -> tuple[object, list[dict]]:
    """Build a BM25 index from a BRIGHT domain docs JSONL file.

    Parameters
    ----------
    docs_path : str
        Path to domain_docs.jsonl file.
    engine : str
        BM25 engine: "rank_bm25" (Python) or "pyserini" (Lucene).
    lucene_index_path : str | None
        Path for Lucene index (required if engine="pyserini").
    pyserini_k1 : float
        BM25 k1 parameter for Pyserini (default 0.9, BRIGHT paper setting).
    pyserini_b : float
        BM25 b parameter for Pyserini (default 0.4, BRIGHT paper setting).

    Returns
    -------
    bm25 : object
        The BM25 index (BM25Okapi or PyseriniBM25).
    docs : list[dict]
        List of {"id": ..., "content": ...} dicts (aligned with index).
    """
    import json as _json

    docs = []
    with open(docs_path) as f:
        for line in f:
            doc = _json.loads(line)
            docs.append(doc)

    if engine == "pyserini":
        set_bm25_pyserini_mode(True)
        from pyserini_bm25 import build_pyserini_index, PyseriniBM25

        if not lucene_index_path:
            # Auto-derive from docs_path
            lucene_index_path = docs_path.replace(".jsonl", "_lucene_index")

        build_pyserini_index(docs_path, lucene_index_path)
        bm25 = PyseriniBM25(lucene_index_path, docs, k1=pyserini_k1, b=pyserini_b)
    else:
        from rank_bm25 import BM25Okapi
        tokenized = [bm25_tokenize(doc["content"]) for doc in docs]
        bm25 = BM25Okapi(tokenized)

    return bm25, docs


# ---------------------------------------------------------------------------
# nDCG computation
# ---------------------------------------------------------------------------

def compute_ndcg_at_k(
    ranked_ids: list[str],
    gold_ids: set[str],
    k: int = 10,
) -> float:
    """Compute nDCG@k for a ranked list of document IDs.

    Uses binary relevance (1 if in gold_ids, 0 otherwise).
    """
    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k]):
        rel = 1.0 if doc_id in gold_ids else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0

    # Ideal DCG
    n_relevant = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_recall_at_k(
    ranked_ids: list[str],
    gold_ids: set[str],
    k: int = 10,
) -> float:
    """Compute Recall@k."""
    if not gold_ids:
        return 0.0
    hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in gold_ids)
    return hits / len(gold_ids)


def compute_mrr(
    ranked_ids: list[str],
    gold_ids: set[str],
) -> float:
    """Compute Mean Reciprocal Rank."""
    for i, doc_id in enumerate(ranked_ids):
        if doc_id in gold_ids:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# BRIGHT Pipeline
# ---------------------------------------------------------------------------

class BrightPipeline:
    """Graph-based re-ranking pipeline for BRIGHT.

    Parameters
    ----------
    initial_top_k : int
        Number of BM25 candidates to retrieve.
    graph_top_k : int
        Number of top BM25 docs to use for graph construction.
    rerank_top_k : int
        Number of documents in final output.
    rerank_alpha : float
        Weight for BM25 score (vs graph score). final = alpha*bm25 + (1-alpha)*graph.
    max_para_freq : int
        Discriminative entity filter for entity graph.
    """

    def __init__(
        self,
        initial_top_k: int = 100,
        graph_top_k: int = 30,
        rerank_top_k: int = 10,
        rerank_alpha: float = 0.5,
        max_para_freq: int = 5,
    ):
        self.initial_top_k = initial_top_k
        self.graph_top_k = graph_top_k
        self.rerank_top_k = rerank_top_k
        self.rerank_alpha = rerank_alpha
        self.max_para_freq = max_para_freq

    def rerank(
        self,
        query: str,
        query_id: str,
        bm25_index: object,
        docs: list[dict],
        gold_ids: set[str] | None = None,
        excluded_ids: set[str] | None = None,
    ) -> BrightResult:
        """Re-rank BM25 results using entity graph topology.

        Parameters
        ----------
        query : str
            The search query.
        query_id : str
            Query identifier.
        bm25_index : BM25Okapi
            Pre-built BM25 index.
        docs : list[dict]
            Documents aligned with the index.
        gold_ids : set[str] | None
            Gold relevant document IDs (for evaluation only).
        excluded_ids : set[str] | None
            Document IDs to exclude from results.

        Returns
        -------
        BrightResult
        """
        t0 = time.time()
        excluded = excluded_ids or set()

        # Phase 1: BM25 retrieval
        query_tokens = bm25_tokenize(query)
        bm25_scores = bm25_index.get_scores(query_tokens)

        # Sort and get top-k (excluding excluded_ids)
        scored = [
            (i, float(bm25_scores[i]))
            for i in range(len(docs))
            if docs[i]["id"] not in excluded
        ]
        scored.sort(key=lambda x: -x[1])
        top_candidates = scored[:self.initial_top_k]

        bm25_doc_ids = [docs[i]["id"] for i, _ in top_candidates]

        # Phase 2: Graph construction from top graph_top_k
        graph_candidates = top_candidates[:self.graph_top_k]

        # Build titles and sentences for entity graph
        titles = []
        sentences_list = []
        doc_id_map = {}  # title -> doc_id

        for idx, (doc_idx, _) in enumerate(graph_candidates):
            doc = docs[doc_idx]
            title = f"doc_{idx}"
            doc_id_map[title] = doc["id"]

            # Split document content into sentences
            content = doc["content"]
            sents = _split_sentences(content, max_sentences=30)
            if not sents:
                sents = [content[:500]]

            titles.append(title)
            sentences_list.append(sents)

        # Build entity graph
        graph = build_sentence_graph(
            titles, sentences_list, max_para_freq=self.max_para_freq
        )

        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()

        # Compute topology
        if n_nodes > 0:
            components = list(nx.connected_components(graph))
            beta_0 = len(components)
            beta_1 = n_edges - n_nodes + beta_0
        else:
            beta_0, beta_1 = 0, 0

        # Phase 3: Compute graph scores
        graph_scores = self._compute_graph_scores(
            query, graph, titles, sentences_list, doc_id_map
        )

        # Phase 4: Combine BM25 + graph scores
        # Normalize BM25 scores
        bm25_max = max(s for _, s in top_candidates) if top_candidates else 1.0
        bm25_min = min(s for _, s in top_candidates) if top_candidates else 0.0
        bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1.0

        combined: list[tuple[str, float]] = []
        for doc_idx, bm25_score in top_candidates:
            doc_id = docs[doc_idx]["id"]
            bm25_norm = (bm25_score - bm25_min) / bm25_range

            g_score = graph_scores.get(doc_id, 0.0)

            final = self.rerank_alpha * bm25_norm + (1 - self.rerank_alpha) * g_score
            combined.append((doc_id, final))

        # Sort by final score
        combined.sort(key=lambda x: -x[1])
        ranked_ids = [doc_id for doc_id, _ in combined[:self.rerank_top_k]]
        ranked_scores = [score for _, score in combined[:self.rerank_top_k]]

        return BrightResult(
            query_id=query_id,
            ranked_doc_ids=ranked_ids,
            ranked_scores=ranked_scores,
            bm25_doc_ids=bm25_doc_ids[:self.rerank_top_k],
            beta_0=beta_0,
            beta_1=beta_1,
            n_graph_nodes=n_nodes,
            n_graph_edges=n_edges,
            n_docs_in_graph=len(graph_candidates),
            latency_ms=(time.time() - t0) * 1000,
        )

    def _compute_graph_scores(
        self,
        query: str,
        graph: nx.Graph,
        titles: list[str],
        sentences_list: list[list[str]],
        doc_id_map: dict[str, str],
    ) -> dict[str, float]:
        """Compute graph-based relevance scores for each document.

        Score components:
          1. PageRank centrality (information hub)
          2. Query entity overlap (direct relevance)
          3. Component connectivity bonus (β₀ contribution)
        """
        if graph.number_of_nodes() == 0:
            return {}

        # PageRank
        try:
            pagerank = nx.pagerank(graph, weight="weight")
        except Exception:
            pagerank = {n: 1.0 / graph.number_of_nodes() for n in graph.nodes()}

        # Query entities
        query_entities = extract_entities(query)
        query_tokens = set(query.lower().split())

        # Aggregate per-document scores
        doc_scores: dict[str, float] = {}

        for title_idx, title in enumerate(titles):
            doc_id = doc_id_map.get(title, title)

            # Collect nodes belonging to this document
            doc_nodes = [
                n for n in graph.nodes()
                if graph.nodes[n].get("title") == title
            ]

            if not doc_nodes:
                doc_scores[doc_id] = 0.0
                continue

            # 1. Average PageRank of document's nodes
            pr_sum = sum(pagerank.get(n, 0.0) for n in doc_nodes)
            pr_avg = pr_sum / len(doc_nodes)

            # 2. Entity overlap with query
            doc_entities = set()
            doc_text_tokens = set()
            for sent in sentences_list[title_idx]:
                doc_entities.update(extract_entities(sent))
                doc_text_tokens.update(sent.lower().split())

            entity_overlap = len(query_entities & doc_entities)
            token_overlap = len(query_tokens & doc_text_tokens)

            # Normalize overlaps
            entity_score = entity_overlap / max(len(query_entities), 1)
            token_score = min(token_overlap / max(len(query_tokens), 1), 1.0)

            # 3. Degree centrality (connectivity)
            degree_sum = sum(graph.degree(n) for n in doc_nodes)
            degree_avg = degree_sum / len(doc_nodes)
            degree_norm = min(degree_avg / 10.0, 1.0)  # normalize

            # Combined graph score
            score = (
                0.4 * pr_avg * graph.number_of_nodes()  # scale PR
                + 0.3 * entity_score
                + 0.2 * token_score
                + 0.1 * degree_norm
            )

            doc_scores[doc_id] = score

        # Normalize graph scores to [0, 1]
        if doc_scores:
            max_gs = max(doc_scores.values())
            if max_gs > 0:
                doc_scores = {k: v / max_gs for k, v in doc_scores.items()}

        return doc_scores


# ---------------------------------------------------------------------------
# Sentence splitting utility
# ---------------------------------------------------------------------------

def _split_sentences(text: str, max_sentences: int = 30) -> list[str]:
    """Split document text into sentences."""
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    raw = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in raw if len(s.strip()) >= 10]
    return sentences[:max_sentences]
