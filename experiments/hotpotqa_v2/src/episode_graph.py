"""Episode-level knowledge graph for geDIG routing.

Builds graphs where nodes are atomic episodes (from LLM decomposition)
rather than sentences. Edges follow the synapse model (binary, unweighted)
with density control via k_target.

Edge types (all binary — geDIG uses edge existence only):
  - Sequential: episode_i → episode_{i+1} within same doc/query
  - Connects_to: LLM-determined logical connections within same doc/query
  - Cross-doc: entity overlap + dense similarity between episodes of different docs
  - Cross-query: query episode → document episode (top-k_target per query ep)

Node features (388D):
  dim 0-383:  E5 dense embedding (semantic content)
  dim 384:    TF-IDF similarity to query
  dim 385:    Entity overlap with query
  dim 386:    BM25 score (normalized)
  dim 387:    Position within document (0.0-1.0)
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Data structures
# ------------------------------------------------------------------ #

@dataclass
class Episode:
    """A single atomic episode."""
    id: int
    text: str
    type: str
    connects_to: list[int] = field(default_factory=list)


@dataclass
class DocEpisodes:
    """All episodes for a single document."""
    doc_id: str
    episodes: list[Episode]
    method: str = "llm"


@dataclass
class EpisodeGraphResult:
    """Result of episode graph construction."""
    g_before: nx.Graph
    g_after: nx.Graph
    features_before: np.ndarray
    features_after: np.ndarray
    focal_nodes: set[str]
    node_order_before: list[str]
    node_order_after: list[str]
    n_doc_episodes: int
    n_query_episodes: int
    n_intra_doc_edges: int
    n_cross_doc_edges: int
    n_query_edges: int


# ------------------------------------------------------------------ #
# Entity extraction (simplified from entity_graph.py)
# ------------------------------------------------------------------ #

_NE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
_STOP_ENTITIES = {
    "the", "a", "an", "he", "she", "it", "they", "his", "her", "its",
    "this", "that", "these", "those", "who", "which", "what",
    "however", "therefore", "although", "because", "also", "many",
    "some", "most", "other", "such", "each", "every", "both",
    "several", "various", "certain", "another", "few", "all",
}


def _extract_entities(text: str) -> set[str]:
    """Extract named entities from text (simple capitalized-word heuristic)."""
    entities = set()
    for match in _NE_RE.finditer(text):
        entity = match.group().lower()
        if entity not in _STOP_ENTITIES and len(entity) > 2:
            entities.add(entity)
    return entities


def _entity_jaccard(text_a: str, text_b: str) -> float:
    """Entity Jaccard similarity between two texts."""
    ents_a = _extract_entities(text_a)
    ents_b = _extract_entities(text_b)
    if not ents_a or not ents_b:
        return 0.0
    intersection = ents_a & ents_b
    union = ents_a | ents_b
    return len(intersection) / len(union) if union else 0.0


# ------------------------------------------------------------------ #
# Episode index (loads pre-computed episodes from disk)
# ------------------------------------------------------------------ #

class EpisodeIndex:
    """Loads and serves pre-computed episode data."""

    def __init__(self, episode_dir: str | Path):
        self.episode_dir = Path(episode_dir)
        self._doc_episodes: dict[str, dict] = {}  # domain -> {doc_id: raw_dict}
        self._query_episodes: dict[str, dict] = {}  # domain -> {query_id: raw_dict}

    def load_domain(self, domain: str) -> None:
        """Load episode data for a domain."""
        # Document episodes
        doc_path = self.episode_dir / f"{domain}_episodes.jsonl"
        if doc_path.exists():
            self._doc_episodes[domain] = {}
            with open(doc_path) as f:
                for line in f:
                    rec = json.loads(line)
                    self._doc_episodes[domain][rec["doc_id"]] = rec
            logger.info("Loaded %d doc episodes for %s",
                        len(self._doc_episodes[domain]), domain)
        else:
            logger.warning("Doc episodes not found: %s", doc_path)
            self._doc_episodes[domain] = {}

        # Query episodes
        query_path = self.episode_dir / f"{domain}_query_episodes.jsonl"
        if query_path.exists():
            self._query_episodes[domain] = {}
            with open(query_path) as f:
                for line in f:
                    rec = json.loads(line)
                    self._query_episodes[domain][rec["query_id"]] = rec
            logger.info("Loaded %d query episodes for %s",
                        len(self._query_episodes[domain]), domain)
        else:
            logger.warning("Query episodes not found: %s", query_path)
            self._query_episodes[domain] = {}

    def get_doc_episodes(
        self, domain: str, doc_ids: list[str]
    ) -> list[DocEpisodes]:
        """Get episodes for specific documents."""
        domain_eps = self._doc_episodes.get(domain, {})
        results = []
        for doc_id in doc_ids:
            rec = domain_eps.get(doc_id)
            if rec is not None:
                episodes = [
                    Episode(
                        id=ep["id"],
                        text=ep["text"],
                        type=ep.get("type", "claim"),
                        connects_to=ep.get("connects_to", []),
                    )
                    for ep in rec["episodes"]
                ]
                results.append(DocEpisodes(
                    doc_id=doc_id,
                    episodes=episodes,
                    method=rec.get("method", "llm"),
                ))
            else:
                # Document not episodified — create single-episode fallback
                results.append(DocEpisodes(
                    doc_id=doc_id,
                    episodes=[Episode(id=0, text="", type="single")],
                    method="missing",
                ))
        return results

    def get_query_episodes(
        self, domain: str, query_id: str
    ) -> list[Episode]:
        """Get episodes for a query."""
        domain_eps = self._query_episodes.get(domain, {})
        rec = domain_eps.get(query_id)
        if rec is not None:
            return [
                Episode(
                    id=ep["id"],
                    text=ep["text"],
                    type=ep.get("type", "question"),
                    connects_to=ep.get("connects_to", []),
                )
                for ep in rec["episodes"]
            ]
        return []


# ------------------------------------------------------------------ #
# Episode graph builder
# ------------------------------------------------------------------ #

class EpisodeGraphBuilder:
    """Builds episode-level graphs for geDIG routing.

    Parameters
    ----------
    k_target : int
        Target number of cross-edges per query episode node.
    dense_retriever : Any
        DenseRetriever instance for embedding lookup.
    dense_domain : str
        Domain for dense retriever.
    feature_dim : int
        Dimension of node feature vectors.
    cross_doc_max_edges : int
        Maximum cross-doc edges to add.
    """

    def __init__(
        self,
        k_target: int = 4,
        dense_retriever: Any = None,
        dense_domain: str = "",
        feature_dim: int = 388,
        cross_doc_max_edges: int = 100,
    ):
        self.k_target = k_target
        self.dense_retriever = dense_retriever
        self.dense_domain = dense_domain
        self.feature_dim = feature_dim
        self.cross_doc_max_edges = cross_doc_max_edges

    def build(
        self,
        doc_episodes_list: list[DocEpisodes],
        query_episodes: list[Episode],
        query_text: str,
        doc_bm25_scores: dict[str, float] | None = None,
    ) -> EpisodeGraphResult:
        """Build episode graph with before/after query injection.

        Parameters
        ----------
        doc_episodes_list : list[DocEpisodes]
            Episodes for each document in the candidate pool.
        query_episodes : list[Episode]
            Episodes decomposed from the query.
        query_text : str
            Original query text (for feature computation).
        doc_bm25_scores : dict[str, float] | None
            BM25 scores per doc_id (for node features).

        Returns
        -------
        EpisodeGraphResult
        """
        g = nx.Graph()
        n_intra = 0
        n_cross = 0

        # ----- Phase 1: Add document episode nodes -----
        doc_node_map: dict[str, list[str]] = {}  # doc_id → [node_ids]

        for doc_ep in doc_episodes_list:
            node_ids = []
            for ep in doc_ep.episodes:
                node_id = f"d_{doc_ep.doc_id}_ep{ep.id}"
                g.add_node(
                    node_id,
                    text=ep.text,
                    ep_type=ep.type,
                    is_query=False,
                    doc_id=doc_ep.doc_id,
                    ep_id=ep.id,
                    total_in_doc=len(doc_ep.episodes),
                )
                node_ids.append(node_id)

                # Sequential edge (within doc)
                if ep.id > 0:
                    prev_id = f"d_{doc_ep.doc_id}_ep{ep.id - 1}"
                    if prev_id in g:
                        g.add_edge(prev_id, node_id)
                        n_intra += 1

                # Connects_to edges (within doc)
                for ref in ep.connects_to:
                    ref_id = f"d_{doc_ep.doc_id}_ep{ref}"
                    if ref_id in g:
                        g.add_edge(ref_id, node_id)
                        n_intra += 1

            doc_node_map[doc_ep.doc_id] = node_ids

        n_doc_episodes = g.number_of_nodes()

        # ----- Phase 2: Cross-document edges (entity + dense) -----
        n_cross = self._add_cross_doc_edges(g, doc_episodes_list, doc_node_map)

        # Snapshot: g_before
        g_before = g.copy()
        node_order_before = list(g_before.nodes())

        # ----- Phase 3: Add query episode nodes -----
        query_node_ids = []
        for ep in query_episodes:
            node_id = f"q_ep{ep.id}"
            g.add_node(
                node_id,
                text=ep.text,
                ep_type=ep.type,
                is_query=True,
                doc_id="__query__",
                ep_id=ep.id,
                total_in_doc=len(query_episodes),
            )
            query_node_ids.append(node_id)

            # Sequential edge (within query)
            if ep.id > 0:
                prev_id = f"q_ep{ep.id - 1}"
                if prev_id in g:
                    g.add_edge(prev_id, node_id)

            # Connects_to edges (within query)
            for ref in ep.connects_to:
                ref_id = f"q_ep{ref}"
                if ref_id in g:
                    g.add_edge(ref_id, node_id)

        # ----- Phase 4: Cross-edges query → doc episodes -----
        n_query_edges = self._add_query_edges(
            g, query_episodes, query_node_ids,
            doc_episodes_list, doc_node_map
        )

        node_order_after = list(g.nodes())
        focal_nodes = set(query_node_ids)

        # ----- Phase 5: Compute features -----
        features_before = self._compute_features(
            g_before, node_order_before, query_text, doc_bm25_scores
        )
        features_after = self._compute_features(
            g, node_order_after, query_text, doc_bm25_scores
        )

        # Store features as node attributes for geDIG to use
        for i, node_id in enumerate(node_order_before):
            g_before.nodes[node_id]["feature"] = features_before[i]
        for i, node_id in enumerate(node_order_after):
            g.nodes[node_id]["feature"] = features_after[i]

        return EpisodeGraphResult(
            g_before=g_before,
            g_after=g,
            features_before=features_before,
            features_after=features_after,
            focal_nodes=focal_nodes,
            node_order_before=node_order_before,
            node_order_after=node_order_after,
            n_doc_episodes=n_doc_episodes,
            n_query_episodes=len(query_episodes),
            n_intra_doc_edges=n_intra,
            n_cross_doc_edges=n_cross,
            n_query_edges=n_query_edges,
        )

    def _add_cross_doc_edges(
        self,
        g: nx.Graph,
        doc_episodes_list: list[DocEpisodes],
        doc_node_map: dict[str, list[str]],
    ) -> int:
        """Add cross-document edges based on entity overlap and dense similarity.

        Uses the synapse model: edges are binary (exist/not exist).
        Density controlled by selecting top-scoring pairs.
        """
        n_edges = 0

        # Build episode pairs across documents
        # Use first episode of each doc as representative
        doc_reps: list[tuple[str, str, str]] = []  # (doc_id, node_id, text)
        for doc_ep in doc_episodes_list:
            if doc_ep.episodes and doc_node_map.get(doc_ep.doc_id):
                first_node = doc_node_map[doc_ep.doc_id][0]
                doc_reps.append((doc_ep.doc_id, first_node, doc_ep.episodes[0].text))

        # Get embeddings for cross-doc similarity
        doc_embeddings: dict[str, np.ndarray] = {}
        if self.dense_retriever is not None and self.dense_domain:
            doc_ids = [dr[0] for dr in doc_reps]
            doc_embeddings = self.dense_retriever.get_doc_embeddings(
                self.dense_domain, doc_ids
            )

        # Score all pairs and keep top-k
        pair_scores: list[tuple[float, str, str]] = []
        for i in range(len(doc_reps)):
            for j in range(i + 1, len(doc_reps)):
                doc_id_a, node_a, text_a = doc_reps[i]
                doc_id_b, node_b, text_b = doc_reps[j]

                score = 0.0
                # Entity overlap
                ent_sim = _entity_jaccard(text_a, text_b)
                if ent_sim > 0:
                    score += 3.0 * ent_sim

                # Dense embedding similarity
                emb_a = doc_embeddings.get(doc_id_a)
                emb_b = doc_embeddings.get(doc_id_b)
                if emb_a is not None and emb_b is not None:
                    cos_sim = float(emb_a @ emb_b)
                    score += max(0, cos_sim)

                if score > 0.3:  # Minimum threshold
                    pair_scores.append((score, node_a, node_b))

        # Sort by score descending and take top-k
        pair_scores.sort(key=lambda x: -x[0])
        for score, node_a, node_b in pair_scores[:self.cross_doc_max_edges]:
            if not g.has_edge(node_a, node_b):
                g.add_edge(node_a, node_b)
                n_edges += 1

        return n_edges

    def _add_query_edges(
        self,
        g: nx.Graph,
        query_episodes: list[Episode],
        query_node_ids: list[str],
        doc_episodes_list: list[DocEpisodes],
        doc_node_map: dict[str, list[str]],
    ) -> int:
        """Add cross-edges from query episodes to document episodes.

        Each query episode gets at most k_target connections to doc episodes.
        """
        n_edges = 0

        # Flatten all doc episode nodes for scoring
        doc_ep_nodes: list[tuple[str, str]] = []  # (node_id, text)
        for doc_ep in doc_episodes_list:
            for ep in doc_ep.episodes:
                node_id = f"d_{doc_ep.doc_id}_ep{ep.id}"
                if node_id in g:
                    doc_ep_nodes.append((node_id, ep.text))

        # Get embeddings for query episodes (encode on the fly)
        query_embs: dict[str, np.ndarray] = {}
        doc_ep_embs: dict[str, np.ndarray] = {}

        if self.dense_retriever is not None:
            # Encode query episode texts
            model = self.dense_retriever.model
            for ep, node_id in zip(query_episodes, query_node_ids):
                if ep.text.strip():
                    emb = model.encode(
                        ["query: " + ep.text[:480]],
                        normalize_embeddings=True,
                    ).astype(np.float32)[0]
                    query_embs[node_id] = emb

            # Get doc episode embeddings (from parent doc embeddings as proxy)
            doc_id_set = set()
            for doc_ep in doc_episodes_list:
                doc_id_set.add(doc_ep.doc_id)
            doc_parent_embs = self.dense_retriever.get_doc_embeddings(
                self.dense_domain, list(doc_id_set)
            )
            for doc_ep in doc_episodes_list:
                parent_emb = doc_parent_embs.get(doc_ep.doc_id)
                if parent_emb is not None:
                    for ep in doc_ep.episodes:
                        node_id = f"d_{doc_ep.doc_id}_ep{ep.id}"
                        doc_ep_embs[node_id] = parent_emb

        # Score and connect
        for q_ep, q_node_id in zip(query_episodes, query_node_ids):
            scores: list[tuple[float, str]] = []

            for doc_node_id, doc_text in doc_ep_nodes:
                score = 0.0

                # Entity overlap
                ent_sim = _entity_jaccard(q_ep.text, doc_text)
                if ent_sim > 0:
                    score += 3.0 * ent_sim

                # Dense similarity
                q_emb = query_embs.get(q_node_id)
                d_emb = doc_ep_embs.get(doc_node_id)
                if q_emb is not None and d_emb is not None:
                    cos_sim = float(q_emb @ d_emb)
                    score += max(0, cos_sim)

                if score > 0.1:
                    scores.append((score, doc_node_id))

            # Top-k_target connections
            scores.sort(key=lambda x: -x[0])
            for score, doc_node_id in scores[:self.k_target]:
                g.add_edge(q_node_id, doc_node_id)
                n_edges += 1

        return n_edges

    def _compute_features(
        self,
        g: nx.Graph,
        node_order: list[str],
        query_text: str,
        doc_bm25_scores: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Compute multi-dimensional node features.

        Returns array of shape (N_nodes, feature_dim).
        Uses simplified features when dense_retriever is unavailable.
        """
        n = len(node_order)
        if n == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        features = np.zeros((n, self.feature_dim), dtype=np.float32)

        # Get embeddings
        has_embeddings = False
        if self.dense_retriever is not None:
            model = self.dense_retriever.model
            # Encode all node texts at once
            texts = []
            for node_id in node_order:
                nd = g.nodes[node_id]
                text = nd.get("text", "")
                if nd.get("is_query", False):
                    texts.append("query: " + text[:480])
                else:
                    texts.append("passage: " + text[:480])

            if texts:
                try:
                    embs = model.encode(
                        texts,
                        batch_size=32,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    ).astype(np.float32)
                    # Fill dims 0-383 with embeddings
                    emb_dim = min(embs.shape[1], 384)
                    features[:, :emb_dim] = embs[:, :emb_dim]
                    has_embeddings = True
                except Exception as e:
                    logger.warning("Embedding error: %s", e)

        if not has_embeddings:
            # Fallback: random features (geDIG will still work on topology)
            rng = np.random.RandomState(42)
            features[:, :384] = rng.randn(n, 384).astype(np.float32) * 0.01

        # Additional features (dims 384-387)
        query_entities = _extract_entities(query_text)

        for i, node_id in enumerate(node_order):
            nd = g.nodes[node_id]
            text = nd.get("text", "")
            doc_id = nd.get("doc_id", "")
            ep_id = nd.get("ep_id", 0)
            total = nd.get("total_in_doc", 1)

            # dim 384: entity overlap with query
            ep_entities = _extract_entities(text)
            if query_entities and ep_entities:
                overlap = len(query_entities & ep_entities) / len(
                    query_entities | ep_entities
                )
                features[i, 384] = overlap

            # dim 385: BM25 score (normalized)
            if doc_bm25_scores and doc_id in doc_bm25_scores:
                features[i, 385] = doc_bm25_scores[doc_id]

            # dim 386: position within document
            features[i, 386] = ep_id / max(total, 1)

            # dim 387: is_query flag
            features[i, 387] = 1.0 if nd.get("is_query", False) else 0.0

        return features
