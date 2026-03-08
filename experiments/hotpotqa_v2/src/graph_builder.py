"""Knowledge graph builder for HotpotQA v2 — beta_0-sensitive design.

Key design change from v1
-------------------------
v1 connected the question node Q to **all** retrieved facts, which
made the graph always connected (beta_0 = 1) and prevented any
beta_0 variation.  For bridge-type questions this is fatal because
the two supporting document islands are never detected.

v2 connects Q to only the **top-k_q** (default 3) facts by BM25
score.  Intra-title facts are linked sequentially (adjacent
sentences).  Cross-title facts are linked only when they share
named entities above a threshold.  This naturally produces
disconnected components (beta_0 > 1) when the bridge fact has not
yet been retrieved.  Adding a bridge fact merges components,
yielding delta_beta_0 = -1 — exactly the signal the extended
gauge needs.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass

import networkx as nx

from .retriever import RetrievedFact

_TOKEN_RE = re.compile(r"[a-z0-9']+")

# Simple named-entity proxy: capitalized word sequences
_NE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")


@dataclass
class GraphBuildConfig:
    """Tuneable parameters for the graph builder."""

    # Number of top BM25 facts connected directly to Q
    q_link_top_k: int = 3
    # TF-IDF feature dimension for GeDIGCore compatibility
    tfidf_dim: int = 64
    # Minimum entity overlap ratio to create a cross-title edge
    entity_overlap_threshold: float = 0.3
    # Weight for Q→fact edges (based on BM25 rank)
    q_edge_base_weight: float = 0.9
    # Weight for same-title adjacent sentence edges
    intra_title_weight: float = 0.8
    # Weight for cross-title entity overlap edges
    cross_title_weight: float = 0.5

    # v5: Two-Edge Architecture
    two_edge_mode: bool = False
    # Context attention: max sent_id distance for intra-title edges
    ctx_max_sent_distance: int = 6
    # Similarity attention: TF-IDF cosine weight
    sim_alpha: float = 0.6
    # Similarity attention: entity overlap weight
    sim_beta: float = 0.4
    # Similarity attention: minimum score to create a cross-title edge
    sim_edge_threshold: float = 0.25


class KnowledgeGraphBuilder:
    """Build GeDIGCore-compatible knowledge graphs from retrieved facts.

    The builder produces ``nx.Graph`` objects whose nodes carry a
    ``feature`` attribute (list[float]) that GeDIGCore can consume for
    entropy calculation.
    """

    def __init__(self, config: GraphBuildConfig | None = None):
        self.cfg = config or GraphBuildConfig()

    # ------------------------------------------------------------------ #
    # Feature vector helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    @staticmethod
    def _stable_hash(token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little", signed=False)

    def build_tfidf_stats(
        self, facts: list[RetrievedFact]
    ) -> tuple[dict[str, float], int]:
        """Build IDF map from a corpus of retrieved facts."""
        if self.cfg.tfidf_dim <= 0:
            return {}, len(facts)
        df: Counter[str] = Counter()
        for fact in facts:
            tokens = self._tokenize(fact.title) + self._tokenize(fact.text)
            for tok in set(tokens):
                df[tok] += 1
        n = max(len(facts), 1)
        idf = {tok: math.log((n + 1) / (freq + 1)) + 1.0 for tok, freq in df.items()}
        return idf, n

    def _tfidf_vector(
        self, tokens: list[str], idf_map: dict[str, float], doc_count: int
    ) -> list[float]:
        if self.cfg.tfidf_dim <= 0:
            return []
        if not tokens:
            return [0.0] * self.cfg.tfidf_dim
        total = len(tokens)
        counts = Counter(tokens)
        default_idf = math.log((doc_count + 1) / 2.0) + 1.0 if doc_count > 0 else 1.0
        vec = [0.0] * self.cfg.tfidf_dim
        for tok, cnt in counts.items():
            tf = cnt / total
            idf_val = idf_map.get(tok, default_idf)
            idx = self._stable_hash(tok) % self.cfg.tfidf_dim
            vec[idx] += tf * idf_val
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    @staticmethod
    def _length_norm(tokens: list[str], scale: int = 40) -> float:
        return min(len(tokens) / float(scale), 1.0) if tokens else 0.0

    def _make_question_vector(
        self,
        question_tokens: list[str],
        tfidf_vector: list[float] | None = None,
    ) -> list[float]:
        base = [
            0.0,  # score_norm
            1.0,  # overlap proxy
            self._length_norm(question_tokens),
            1.0,  # title overlap proxy
            1.0,  # is_question
        ]
        if tfidf_vector:
            return base + tfidf_vector
        return base

    def _make_fact_vector(
        self,
        question_tokens: set[str],
        title: str,
        text: str,
        score: float,
        max_score: float,
        tfidf_vector: list[float] | None = None,
    ) -> list[float]:
        text_tokens = self._tokenize(text)
        title_tokens = self._tokenize(title)
        overlap = (
            len(set(text_tokens) & question_tokens) / max(len(question_tokens), 1)
            if question_tokens
            else 0.0
        )
        title_overlap = 1.0 if question_tokens and (set(title_tokens) & question_tokens) else 0.0
        score_norm = score / max_score if max_score > 0 else 0.0
        base = [score_norm, overlap, self._length_norm(text_tokens), title_overlap, 0.0]
        if tfidf_vector:
            return base + tfidf_vector
        return base

    # ------------------------------------------------------------------ #
    # Similarity helpers (v5)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot_val = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return dot_val / (norm_a * norm_b)

    # ------------------------------------------------------------------ #
    # Named entity extraction (lightweight heuristic)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_entities(text: str) -> set[str]:
        """Extract capitalised word sequences as a simple NE proxy."""
        return {ent.lower() for ent in _NE_RE.findall(text)}

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #

    def build_empty_graph(
        self,
        question: str,
        idf_map: dict[str, float] | None = None,
        doc_count: int = 1,
    ) -> nx.Graph:
        """Build a graph with only the question node (g_prev baseline)."""
        q_tokens = self._tokenize(question)
        q_tfidf = self._tfidf_vector(q_tokens, idf_map or {}, doc_count)
        g = nx.Graph()
        g.add_node(
            "Q",
            type="question",
            text=question[:100],
            feature=self._make_question_vector(q_tokens, q_tfidf),
        )
        return g

    def build_graph(
        self,
        question: str,
        facts: list[RetrievedFact],
        idf_map: dict[str, float] | None = None,
        doc_count: int | None = None,
    ) -> nx.Graph:
        """Build a knowledge graph from retrieved facts.

        Graph structure (v2 — beta_0 sensitive):
        - Q node connected to top ``q_link_top_k`` facts only
        - Same-title facts connected sequentially (adjacent sent_id)
        - Cross-title facts connected via entity overlap gate

        Parameters
        ----------
        question : str
        facts : list[RetrievedFact]
            Already sorted by descending BM25 score.
        idf_map, doc_count : optional TF-IDF stats (built if None).
        """
        if idf_map is None or doc_count is None:
            idf_map, doc_count = self.build_tfidf_stats(facts)

        q_tokens = self._tokenize(question)
        q_token_set = set(q_tokens)
        q_tfidf = self._tfidf_vector(q_tokens, idf_map, doc_count)

        g = nx.Graph()
        g.add_node(
            "Q",
            type="question",
            text=question[:100],
            feature=self._make_question_vector(q_tokens, q_tfidf),
        )

        max_score = max((f.score for f in facts), default=1.0)

        # --- Add fact nodes + collect TF-IDF vectors ---
        tfidf_per_fact: dict[int, list[float]] = {}
        for i, fact in enumerate(facts):
            fact_tokens = self._tokenize(fact.title) + self._tokenize(fact.text)
            tfidf_vec = self._tfidf_vector(fact_tokens, idf_map, doc_count)
            tfidf_per_fact[i] = tfidf_vec
            node_id = f"F{i}"
            g.add_node(
                node_id,
                type="fact",
                title=fact.title,
                sent_id=fact.sent_id,
                text=fact.text[:100],
                feature=self._make_fact_vector(
                    q_token_set, fact.title, fact.text, fact.score, max_score, tfidf_vec
                ),
            )

        # --- Q → top-k_q edges only (KEY DIFFERENCE from v1) ---
        for i in range(min(self.cfg.q_link_top_k, len(facts))):
            rank_weight = self.cfg.q_edge_base_weight * (1.0 - 0.1 * i)
            g.add_edge("Q", f"F{i}", weight=max(rank_weight, 0.1))

        # --- Title groups (shared by both modes) ---
        title_groups: dict[str, list[int]] = {}
        for i, fact in enumerate(facts):
            title_groups.setdefault(fact.title, []).append(i)

        # --- Entity extraction (shared by both modes) ---
        entities_per_fact: dict[int, set[str]] = {}
        for i, fact in enumerate(facts):
            entities_per_fact[i] = self._extract_entities(fact.text) | self._extract_entities(fact.title)

        titles = [f.title for f in facts]

        if self.cfg.two_edge_mode:
            # ============================================================ #
            # v5 Two-Edge Architecture
            # ============================================================ #

            # --- Context attention edges (intra-title, distance-decay) ---
            for title, indices in title_groups.items():
                indices_sorted = sorted(indices, key=lambda idx: facts[idx].sent_id)
                for a_pos in range(len(indices_sorted)):
                    a_idx = indices_sorted[a_pos]
                    for b_pos in range(a_pos + 1, len(indices_sorted)):
                        b_idx = indices_sorted[b_pos]
                        sent_dist = abs(facts[a_idx].sent_id - facts[b_idx].sent_id)
                        if sent_dist > self.cfg.ctx_max_sent_distance:
                            break  # sorted by sent_id; all further are farther
                        # Distance-decay weight
                        if sent_dist <= 1:
                            w_ctx = 0.9
                        elif sent_dist <= 3:
                            w_ctx = 0.6
                        else:  # 4-6
                            w_ctx = 0.3
                        g.add_edge(
                            f"F{a_idx}", f"F{b_idx}",
                            weight=w_ctx, edge_type="context",
                            w_ctx=w_ctx, w_sim=0.0,
                            strength=w_ctx,
                        )

            # --- Similarity attention edges (cross-title, TF-IDF cosine + entity overlap) ---
            for i in range(len(facts)):
                for j in range(i + 1, len(facts)):
                    if titles[i] == titles[j]:
                        continue  # handled by context attention above

                    # TF-IDF cosine similarity
                    cos_sim = self._cosine_similarity(tfidf_per_fact[i], tfidf_per_fact[j])

                    # Entity overlap
                    ent_i = entities_per_fact[i]
                    ent_j = entities_per_fact[j]
                    ent_overlap = 0.0
                    if ent_i and ent_j:
                        ent_overlap = len(ent_i & ent_j) / min(len(ent_i), len(ent_j))

                    # Combined similarity score
                    w_sim = (self.cfg.sim_alpha * max(cos_sim, 0.0)
                             + self.cfg.sim_beta * ent_overlap)

                    if w_sim >= self.cfg.sim_edge_threshold:
                        g.add_edge(
                            f"F{i}", f"F{j}",
                            weight=w_sim, edge_type="similarity",
                            w_ctx=0.0, w_sim=w_sim,
                            strength=w_sim,
                        )
        else:
            # ============================================================ #
            # Legacy edge construction (v2-v4, backward compatible)
            # ============================================================ #

            # --- Intra-title edges (same title, adjacent sent_id) ---
            for title, indices in title_groups.items():
                indices_sorted = sorted(indices, key=lambda idx: facts[idx].sent_id)
                for a, b in zip(indices_sorted, indices_sorted[1:]):
                    if abs(facts[a].sent_id - facts[b].sent_id) <= 1:
                        g.add_edge(f"F{a}", f"F{b}", weight=self.cfg.intra_title_weight)

            # --- Cross-title edges (entity overlap gate) ---
            for i in range(len(facts)):
                for j in range(i + 1, len(facts)):
                    if titles[i] == titles[j]:
                        continue
                    ent_i = entities_per_fact[i]
                    ent_j = entities_per_fact[j]
                    if not ent_i or not ent_j:
                        continue
                    overlap = len(ent_i & ent_j) / min(len(ent_i), len(ent_j))
                    if overlap >= self.cfg.entity_overlap_threshold:
                        g.add_edge(f"F{i}", f"F{j}", weight=self.cfg.cross_title_weight)

        return g

    def build_incremental(
        self,
        g_prev: nx.Graph,
        question: str,
        new_facts: list[RetrievedFact],
        existing_facts: list[RetrievedFact],
        idf_map: dict[str, float] | None = None,
        doc_count: int | None = None,
    ) -> nx.Graph:
        """Build an expanded graph by adding new facts to an existing graph.

        Returns a new graph (does not mutate ``g_prev``).
        """
        all_facts = list(existing_facts) + list(new_facts)
        return self.build_graph(question, all_facts, idf_map, doc_count)

    def get_query_vector(
        self,
        question: str,
        idf_map: dict[str, float] | None = None,
        doc_count: int = 1,
    ) -> list[float]:
        """Return the query feature vector (for GeDIGCore compatibility)."""
        q_tokens = self._tokenize(question)
        q_tfidf = self._tfidf_vector(q_tokens, idf_map or {}, doc_count)
        return self._make_question_vector(q_tokens, q_tfidf)
