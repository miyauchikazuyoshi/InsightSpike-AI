"""v10c: Sentence-Level Three-Tier Entity Graph for Multi-hop QA.

Architecture modeled after graph_builder.py v5 Two-Edge design, extended
to three tiers with sentence-level granularity.

Three-Tier Edge Hierarchy (separate types, clear priority):
  Tier 1 — Context (certain, cost 0.05-0.10):
    - Same-paragraph adjacent sentences with distance-decay
      (dist=1: cost=0.05, dist=2-3: cost=0.08, dist=4-6: cost=0.10)
    - Title cross-reference (sentence mentions another paragraph's title):
      cost=0.08

  Tier 2 — Entity (estimated, cost 0.20-0.50):
    - Cross-paragraph discriminative entity overlap (freq <= max_para_freq)
    - cost = 0.20 + 0.30 * (1 - overlap_ratio)

  Tier 3 — Similarity (weak, cost 0.50-0.80):
    - Cross-paragraph TF-IDF cosine similarity (threshold >= 0.30)
    - cost = 0.80 - 0.30 * cosine_sim

Key improvements over v10b (flat mixing):
  1. Edge types are SEPARATE with clear priority hierarchy
  2. Weighted shortest path naturally prefers Tier 1 > Tier 2 > Tier 3
  3. Context attention (same-paragraph) is ALWAYS stronger than cross-paragraph
  4. Sentence-level nodes provide finer-grained connections
  5. No noisy shortcuts from flat signal mixing (v10b's beta_1 problem)
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# 1a. Entity extraction with stopword filtering
# ---------------------------------------------------------------------------

_NE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

# Common words that happen to be capitalised at sentence start or in titles
_STOP_ENTITIES: set[str] = {
    # Pronouns / determiners
    "the", "a", "an", "he", "she", "it", "they", "his", "her", "its",
    "this", "that", "these", "those", "who", "which", "what", "where",
    "when", "how", "why", "whom", "whose",
    # Conjunctions / prepositions / adverbs
    "and", "but", "or", "nor", "for", "yet", "so", "not",
    "after", "before", "during", "between", "among", "about",
    "also", "however", "although", "because", "since", "while",
    "from", "into", "with", "without", "under", "over", "through",
    # Ordinals / adjectives too generic
    "first", "second", "third", "fourth", "fifth", "last",
    "new", "old", "great", "good", "bad", "big", "small", "long",
    "other", "many", "most", "more", "some", "all", "both", "each",
    "early", "late", "former", "latter", "next", "same",
    # Nationality adjectives (frequently capitalised, low discriminative value)
    "american", "british", "french", "german", "english", "italian",
    "spanish", "russian", "chinese", "japanese", "canadian", "australian",
    "indian", "european", "african", "asian", "irish", "scottish",
    "dutch", "swedish", "norwegian", "danish", "polish", "mexican",
    "brazilian", "korean", "turkish", "greek", "roman", "arab",
    "portuguese", "swiss", "belgian", "austrian", "hungarian",
    "czech", "finnish", "thai", "vietnamese", "indonesian",
    "jewish", "christian", "muslim", "catholic", "protestant",
    # Common verbs that appear capitalised at sentence start
    "was", "were", "has", "have", "had", "been", "being",
    "did", "does", "are", "will", "would", "could", "should",
    "may", "might", "can", "shall", "must",
    "said", "made", "took", "gave", "got", "went", "came", "saw",
    "became", "left", "found", "called", "used", "known", "named",
    # Common nouns too generic
    "born", "died", "year", "years", "time", "part", "end",
    "war", "world", "state", "city", "country", "town", "area",
    "school", "university", "college", "film", "song", "album",
    "book", "show", "series", "season", "episode", "band", "group",
    "team", "game", "league", "cup", "award", "prize",
    "north", "south", "east", "west", "central", "western", "eastern",
    "northern", "southern", "united", "national", "international",
    "general", "major", "minor", "royal", "saint", "mount",
    # Temporal
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday",
    # Other frequently noisy
    "according", "along", "away", "back", "down", "even",
    "just", "like", "near", "off", "only", "still", "then",
    "there", "here", "very", "well", "much", "such",
    "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "several", "various",
    "later", "around", "during", "following", "including",
}


def extract_entities(text: str) -> set[str]:
    """Extract named-entity candidates from text.

    Uses a capitalised-word regex with aggressive stopword filtering.
    Returns lowercased entity strings.
    """
    raw = _NE_RE.findall(text)
    return {
        e.lower()
        for e in raw
        if e.lower() not in _STOP_ENTITIES and len(e) > 2
    }


# ---------------------------------------------------------------------------
# TF-IDF helpers (lightweight, no sklearn dependency)
# ---------------------------------------------------------------------------

_TFIDF_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very",
    "and", "but", "or", "if", "while", "because", "until", "about",
    "also", "just", "up", "down",
}


def _tokenize_for_tfidf(text: str) -> list[str]:
    """Simple word tokenizer for TF-IDF."""
    return [
        w for w in re.findall(r"[a-z]+", text.lower())
        if w not in _TFIDF_STOP and len(w) > 2
    ]


def _compute_tfidf_vectors(texts: list[str]) -> list[dict[str, float]]:
    """Compute TF-IDF sparse vectors for a list of texts."""
    tokenized = [_tokenize_for_tfidf(t) for t in texts]
    n_docs = len(texts)

    # Document frequency
    df: dict[str, int] = defaultdict(int)
    for tokens in tokenized:
        for w in set(tokens):
            df[w] += 1

    # TF-IDF per document (log-normalized TF, smooth IDF)
    vectors: list[dict[str, float]] = []
    for tokens in tokenized:
        if not tokens:
            vectors.append({})
            continue
        tf_counts: dict[str, int] = defaultdict(int)
        for w in tokens:
            tf_counts[w] += 1
        vec: dict[str, float] = {}
        for w, cnt in tf_counts.items():
            tf = 1.0 + math.log(cnt)
            idf = math.log((n_docs + 1) / (df[w] + 1)) + 1.0
            vec[w] = tf * idf
        vectors.append(vec)

    return vectors


def _cosine_sim(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# TF-IDF dense feature matrix for geDIG scoring (Spec H)
# ---------------------------------------------------------------------------

def compute_node_tfidf_features(
    graph: nx.Graph,
    max_features: int = 500,
) -> tuple[np.ndarray, Any]:
    """Compute TF-IDF dense feature vectors for all graph nodes.

    Uses the same sparse TF-IDF approach as Tier 3, but converts to a dense
    matrix suitable for message passing and geDIG scoring.

    Args:
        graph: Entity graph with ``text`` attributes on nodes.
        max_features: Maximum vocabulary size (limits dimensionality).

    Returns:
        features: (N, D) float32 matrix (N = number of nodes, D <= max_features).
        vectorizer: Object with a ``transform(texts) -> (M, D) ndarray`` method
            so the caller can project new texts (e.g. query) into the same space.
    """
    node_list = sorted(graph.nodes())
    texts = [graph.nodes[n].get("text", "") for n in node_list]

    # Build vocabulary from node texts using existing tokenizer
    tokenized = [_tokenize_for_tfidf(t) for t in texts]
    n_docs = len(texts)

    # Document frequency
    df: dict[str, int] = defaultdict(int)
    for tokens in tokenized:
        for w in set(tokens):
            df[w] += 1

    # Select top-K features by document frequency (skip very rare ones)
    feature_scores = {w: cnt for w, cnt in df.items() if cnt >= 2}
    sorted_features = sorted(feature_scores, key=feature_scores.get, reverse=True)
    vocab = sorted_features[:max_features]
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    D = len(vocab)

    if D == 0:
        # Fallback: if no features, return zeros
        features = np.zeros((n_docs, 1), dtype=np.float32)
        return features, _TfidfDenseVectorizer(word_to_idx, df, n_docs, D)

    # Build dense TF-IDF matrix
    features = np.zeros((n_docs, D), dtype=np.float32)
    for doc_i, tokens in enumerate(tokenized):
        if not tokens:
            continue
        tf_counts: dict[str, int] = defaultdict(int)
        for w in tokens:
            tf_counts[w] += 1
        for w, cnt in tf_counts.items():
            idx = word_to_idx.get(w)
            if idx is not None:
                tf = 1.0 + math.log(cnt)
                idf = math.log((n_docs + 1) / (df[w] + 1)) + 1.0
                features[doc_i, idx] = tf * idf

    return features, _TfidfDenseVectorizer(word_to_idx, df, n_docs, D)


class _TfidfDenseVectorizer:
    """Lightweight vectorizer that transforms new texts into the same TF-IDF space."""

    def __init__(self, word_to_idx: dict[str, int], df: dict[str, int],
                 n_docs: int, dim: int):
        self._w2i = word_to_idx
        self._df = df
        self._n_docs = n_docs
        self._dim = dim

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts into (M, D) dense TF-IDF matrix."""
        D = max(self._dim, 1)
        result = np.zeros((len(texts), D), dtype=np.float32)
        for ti, text in enumerate(texts):
            tokens = _tokenize_for_tfidf(text)
            if not tokens:
                continue
            tf_counts: dict[str, int] = defaultdict(int)
            for w in tokens:
                tf_counts[w] += 1
            for w, cnt in tf_counts.items():
                idx = self._w2i.get(w)
                if idx is not None:
                    tf = 1.0 + math.log(cnt)
                    idf = math.log((self._n_docs + 1) / (self._df.get(w, 0) + 1)) + 1.0
                    result[ti, idx] = tf * idf
        return result


# ---------------------------------------------------------------------------
# Sentence-level Three-Tier graph construction
# ---------------------------------------------------------------------------

# Tier 1: Context attention (same-paragraph + title cross-ref)
_CTX_COST_ADJ = 0.05       # adjacent sentences (sent_dist = 1)
_CTX_COST_NEAR = 0.08      # nearby sentences (sent_dist = 2-3)
_CTX_COST_FAR = 0.10       # distant same-paragraph (sent_dist = 4-6)
_CTX_MAX_SENT_DIST = 6     # max distance for context edges
_CTX_COST_TITLEREF = 0.08  # title cross-reference

# Tier 2: Entity overlap (cross-paragraph)
_ENT_COST_MIN = 0.20       # full entity overlap (ratio=1.0)
_ENT_COST_MAX = 0.50       # minimal entity overlap (ratio→0)

# Tier D: Dense embedding similarity (cross-paragraph)
_DENSE_COST_MIN = 0.15     # high cosine → near Tier 2 top
_DENSE_COST_MAX = 0.45     # threshold cosine → near Tier 2 bottom

# Tier 3: TF-IDF cosine similarity (cross-paragraph)
_SIM_THRESHOLD = 0.30      # minimum cosine similarity
_SIM_COST_MIN = 0.50       # high cosine (sim=1.0)
_SIM_COST_MAX = 0.80       # threshold cosine (sim=0.30)


def build_sentence_graph(
    titles: list[str],
    sentences_list: list[list[str]],
    max_para_freq: int = 3,
    doc_embeddings: dict[str, np.ndarray] | None = None,
    dense_sim_threshold: float = 0.5,
) -> nx.Graph:
    """Build a sentence-level graph with Three-Tier edge hierarchy.

    Each sentence becomes a node. Edges are created in three tiers with
    non-overlapping cost ranges, ensuring clear priority:
      Tier 1 (context):    cost 0.05 - 0.10  (same-paragraph + title cross-ref)
      Tier D (dense):      cost 0.15 - 0.45  (cross-paragraph dense embedding)
      Tier 2 (entity):     cost 0.20 - 0.50  (cross-paragraph entity overlap)
      Tier 3 (similarity): cost 0.50 - 0.80  (cross-paragraph TF-IDF cosine)

    Higher-tier edges are never overwritten by lower-tier edges.

    Parameters
    ----------
    titles : list[str]
        Paragraph titles (one per paragraph).
    sentences_list : list[list[str]]
        Sentences per paragraph.
    max_para_freq : int
        Maximum paragraph frequency for discriminative entity filtering.
    doc_embeddings : dict[str, np.ndarray] | None
        Pre-computed dense embeddings per paragraph title (from DenseRetriever).
        When provided, Tier D (dense similarity) edges are added.
    dense_sim_threshold : float
        Minimum cosine similarity for Tier D edges.

    Returns
    -------
    nx.Graph
        Sentence-level graph with edge attributes: edge_type, cost, strength.
    """
    n_paras = len(titles)

    # --- Flatten sentences into node list ---
    # Each node: (flat_idx, para_idx, sent_idx, text, title)
    nodes: list[tuple[int, int, int, str, str]] = []
    para_to_nodes: dict[int, list[int]] = defaultdict(list)

    flat_idx = 0
    for p_idx in range(n_paras):
        title = titles[p_idx]
        sents = sentences_list[p_idx]
        for s_idx, sent in enumerate(sents):
            nodes.append((flat_idx, p_idx, s_idx, sent, title))
            para_to_nodes[p_idx].append(flat_idx)
            flat_idx += 1

    n_nodes = len(nodes)

    # --- Build graph with sentence nodes ---
    g = nx.Graph()
    for fi, p_idx, s_idx, text, title in nodes:
        g.add_node(fi, para_idx=p_idx, sent_idx=s_idx, title=title,
                   text=text[:100])

    # --- Extract entities per sentence ---
    sent_entities: list[set[str]] = [
        extract_entities(nodes[i][3]) for i in range(n_nodes)
    ]

    # --- Discriminative entity filtering (by paragraph frequency) ---
    entity_para_freq: dict[str, set[int]] = defaultdict(set)
    for fi, p_idx, _, _, _ in nodes:
        for e in sent_entities[fi]:
            entity_para_freq[e].add(p_idx)

    disc_entities: list[set[str]] = [
        {e for e in ents if len(entity_para_freq[e]) <= max_para_freq}
        for ents in sent_entities
    ]

    # --- TF-IDF vectors per sentence ---
    sent_texts = [n[3] for n in nodes]
    tfidf_vecs = _compute_tfidf_vectors(sent_texts)

    # --- Title data for cross-ref ---
    titles_lower = [t.lower() for t in titles]

    # Counters for diagnostics
    n_tier1 = 0
    n_tier_d = 0
    n_tier2 = 0
    n_tier3 = 0

    # ================================================================
    # Tier 1: Context attention (same-paragraph + title cross-ref)
    # ================================================================
    for p_idx, node_indices in para_to_nodes.items():
        # Sort by sent_idx
        sorted_indices = sorted(node_indices, key=lambda fi: nodes[fi][2])
        for a_pos in range(len(sorted_indices)):
            a_fi = sorted_indices[a_pos]
            for b_pos in range(a_pos + 1, len(sorted_indices)):
                b_fi = sorted_indices[b_pos]
                sent_dist = abs(nodes[a_fi][2] - nodes[b_fi][2])
                if sent_dist > _CTX_MAX_SENT_DIST:
                    break  # sorted by sent_idx; all further are farther
                # Distance-decay cost (from graph_builder.py v5)
                if sent_dist <= 1:
                    cost = _CTX_COST_ADJ
                elif sent_dist <= 3:
                    cost = _CTX_COST_NEAR
                else:  # 4-6
                    cost = _CTX_COST_FAR
                g.add_edge(
                    a_fi, b_fi,
                    edge_type="context",
                    cost=cost,
                    strength=1.0 - cost,
                )
                n_tier1 += 1

    # Title cross-reference (Tier 1): sentence mentions another paragraph's title
    for fi in range(n_nodes):
        p_i = nodes[fi][1]
        text_lower = nodes[fi][3].lower()
        for p_j in range(n_paras):
            if p_i == p_j:
                continue
            # Title must be > 3 chars to avoid false positives
            if len(titles_lower[p_j]) <= 3:
                continue
            if titles_lower[p_j] in text_lower:
                # Connect to first sentence of referenced paragraph as anchor
                # (context edges within that paragraph provide further connectivity)
                anchor = para_to_nodes[p_j][0] if para_to_nodes[p_j] else None
                if anchor is not None and not g.has_edge(fi, anchor):
                    g.add_edge(
                        fi, anchor,
                        edge_type="context",
                        cost=_CTX_COST_TITLEREF,
                        strength=1.0 - _CTX_COST_TITLEREF,
                        title_ref=True,
                    )
                    n_tier1 += 1

    # ================================================================
    # Tier D: Dense embedding similarity (cross-paragraph)
    # ================================================================
    if doc_embeddings is not None:
        for pi in range(n_paras):
            emb_i = doc_embeddings.get(titles[pi])
            if emb_i is None:
                continue
            for pj in range(pi + 1, n_paras):
                emb_j = doc_embeddings.get(titles[pj])
                if emb_j is None:
                    continue
                # Cosine similarity (embeddings are L2-normalized → dot = cosine)
                cos_sim = float(emb_i @ emb_j)
                if cos_sim < dense_sim_threshold:
                    continue
                # Cost: 0.15 (cos=1.0) to 0.45 (cos=threshold)
                denom = 1.0 - dense_sim_threshold
                if denom > 0:
                    cost = _DENSE_COST_MAX - (_DENSE_COST_MAX - _DENSE_COST_MIN) * (
                        (cos_sim - dense_sim_threshold) / denom
                    )
                else:
                    cost = _DENSE_COST_MIN
                # Connect first sentence (anchor) of each paragraph
                anchor_i = para_to_nodes[pi][0] if para_to_nodes[pi] else None
                anchor_j = para_to_nodes[pj][0] if para_to_nodes[pj] else None
                if anchor_i is not None and anchor_j is not None:
                    # Only add if no existing edge or existing cost is higher
                    if not g.has_edge(anchor_i, anchor_j) or \
                       g[anchor_i][anchor_j].get("cost", 1.0) > cost:
                        g.add_edge(
                            anchor_i, anchor_j,
                            edge_type="dense_similarity",
                            cost=cost,
                            strength=1.0 - cost,
                            dense_cos_sim=round(cos_sim, 3),
                        )
                        n_tier_d += 1

    # ================================================================
    # Tier 2: Entity overlap (cross-paragraph only)
    # ================================================================
    for i in range(n_nodes):
        p_i = nodes[i][1]
        if not disc_entities[i]:
            continue
        for j in range(i + 1, n_nodes):
            p_j = nodes[j][1]
            if p_i == p_j:
                continue  # same paragraph → handled by Tier 1
            if not disc_entities[j]:
                continue
            shared = disc_entities[i] & disc_entities[j]
            if not shared:
                continue
            min_count = min(len(disc_entities[i]), len(disc_entities[j]))
            if min_count == 0:
                continue
            ratio = len(shared) / min_count
            # cost: 0.20 (full overlap) to 0.50 (minimal overlap)
            cost = _ENT_COST_MIN + (_ENT_COST_MAX - _ENT_COST_MIN) * (1.0 - ratio)
            # Don't overwrite stronger Tier 1 edges
            if g.has_edge(i, j):
                existing_cost = g[i][j].get("cost", 1.0)
                if cost >= existing_cost:
                    continue
            g.add_edge(
                i, j,
                edge_type="entity",
                cost=cost,
                strength=1.0 - cost,
                entities=shared,
                ent_ratio=round(ratio, 3),
            )
            n_tier2 += 1

    # ================================================================
    # Tier 3: TF-IDF cosine similarity (cross-paragraph only)
    # ================================================================
    for i in range(n_nodes):
        p_i = nodes[i][1]
        if not tfidf_vecs[i]:
            continue
        for j in range(i + 1, n_nodes):
            p_j = nodes[j][1]
            if p_i == p_j:
                continue
            if not tfidf_vecs[j]:
                continue
            cos_sim = _cosine_sim(tfidf_vecs[i], tfidf_vecs[j])
            if cos_sim < _SIM_THRESHOLD:
                continue
            # cost: 0.50 (perfect sim) to ~0.71 (threshold sim)
            cost = _SIM_COST_MAX - (_SIM_COST_MAX - _SIM_COST_MIN) * cos_sim
            # Don't overwrite stronger Tier 1/2 edges
            if g.has_edge(i, j):
                existing_cost = g[i][j].get("cost", 1.0)
                if cost >= existing_cost:
                    continue
            g.add_edge(
                i, j,
                edge_type="similarity",
                cost=cost,
                strength=1.0 - cost,
                cos_sim=round(cos_sim, 3),
            )
            n_tier3 += 1

    # Store tier counts as graph-level attributes
    g.graph["n_tier1_edges"] = n_tier1
    g.graph["n_tier_d_edges"] = n_tier_d
    g.graph["n_tier2_edges"] = n_tier2
    g.graph["n_tier3_edges"] = n_tier3

    return g


# ---------------------------------------------------------------------------
# Q-relevant paragraph detection (paragraph-level)
# ---------------------------------------------------------------------------

def _find_q_relevant_paragraphs(
    question: str,
    titles: list[str],
    para_entities: list[set[str]],
) -> list[int]:
    """Identify paragraphs relevant to the question.

    Three-tier detection:
      1. Strong: Title appears in question, OR title entities overlap with Q entities
      2. Medium: Shares >= 2 entities with question
      3. Weak (fallback): Partial title word match (2+ title words in question)

    If no paragraphs found after all tiers, falls back to top-2 by entity
    overlap count to guarantee at least some Q-relevant paragraphs.
    """
    q_lower = question.lower()
    q_entities = extract_entities(question)
    q_words = set(re.findall(r"[a-z]+", q_lower))

    relevant: list[int] = []
    scores: list[tuple[int, float]] = []  # (para_idx, score) for fallback

    for i, (title, ents) in enumerate(zip(titles, para_entities)):
        # Track entity overlap for fallback scoring
        overlap = ents & q_entities
        scores.append((i, len(overlap)))

        # Tier 1 (strong): full title match
        if len(title) > 3 and title.lower() in q_lower:
            relevant.append(i)
            continue

        # Tier 1 (strong): title entities overlap with Q entities
        title_ents = extract_entities(title)
        if title_ents & q_entities:
            relevant.append(i)
            continue

        # Tier 2 (medium): 2+ entity overlap
        if len(overlap) >= 2:
            relevant.append(i)
            continue

        # Tier 3 (weak): partial title word match
        # At least 2 non-trivial title words appear in question
        title_words = set(re.findall(r"[a-z]+", title.lower()))
        title_words -= {"the", "a", "an", "of", "in", "on", "at", "to", "and",
                        "or", "for", "by", "with", "from", "is", "was", "are"}
        matched = title_words & q_words
        if len(title_words) >= 2 and len(matched) >= 2:
            relevant.append(i)

    # Fallback: if no Q-relevant found, pick top-2 by entity overlap
    if not relevant:
        scores.sort(key=lambda x: x[1], reverse=True)
        for idx, score in scores[:2]:
            if score >= 1:
                relevant.append(idx)

    return relevant


# ---------------------------------------------------------------------------
# Reasoning chain extraction (sentence-level → paragraph-level output)
# ---------------------------------------------------------------------------

def extract_reasoning_chain(
    question: str,
    titles: list[str],
    sentences_list: list[list[str]],
    max_para_freq: int = 3,
) -> dict[str, Any]:
    """Extract the multi-hop reasoning chain from the sentence-level graph.

    Internally operates at sentence level for finer-grained path finding,
    then maps the result back to paragraph indices for the reasoning guide.

    Returns
    -------
    dict with keys:
        chain : list[int]
            Paragraph indices forming the reasoning chain.
        bridge_entities : list[dict]
            Each dict: {"from": title, "to": title, "via": [entity_strings]}.
        topology : dict
            Graph topology features.
        chain_found : bool
            Whether a meaningful chain was found.
    """
    g = build_sentence_graph(titles, sentences_list, max_para_freq)

    # Extract entities per paragraph for Q-relevance check
    para_entities: list[set[str]] = []
    for title, sents in zip(titles, sentences_list):
        full_text = title + " " + " ".join(sents)
        para_entities.append(extract_entities(full_text))

    # Find Q-relevant paragraphs
    q_relevant = _find_q_relevant_paragraphs(question, titles, para_entities)

    # --- Build para_to_nodes mapping from graph ---
    para_to_nodes: dict[int, list[int]] = defaultdict(list)
    for node in g.nodes:
        para_to_nodes[g.nodes[node]["para_idx"]].append(node)

    # --- Topology features ---
    beta_0 = nx.number_connected_components(g)
    beta_1 = g.number_of_edges() - g.number_of_nodes() + beta_0

    try:
        graph_bridges = list(nx.bridges(g))
    except nx.NetworkXError:
        graph_bridges = []

    topology = {
        "beta_0": beta_0,
        "beta_1": beta_1,
        "n_bridges": len(graph_bridges),
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "n_q_relevant": len(q_relevant),
        # v10c-specific: sentence-level and tier diagnostics
        "n_sent_nodes": g.number_of_nodes(),
        "n_tier1_edges": g.graph.get("n_tier1_edges", 0),
        "n_tier_d_edges": g.graph.get("n_tier_d_edges", 0),
        "n_tier2_edges": g.graph.get("n_tier2_edges", 0),
        "n_tier3_edges": g.graph.get("n_tier3_edges", 0),
    }

    # --- Find reasoning chain at sentence level ---
    chain: list[int] = []
    bridge_entities: list[dict[str, Any]] = []
    chain_found = False
    best_sent_path: list[int] | None = None

    if len(q_relevant) >= 2:
        # Try all pairs of Q-relevant paragraphs.
        # For each pair, find shortest sentence-level path.
        best_cost = float("inf")

        for i_idx in range(len(q_relevant)):
            for j_idx in range(i_idx + 1, len(q_relevant)):
                src_para = q_relevant[i_idx]
                dst_para = q_relevant[j_idx]
                # Try all sentence pairs between the two paragraphs
                for src_sent in para_to_nodes[src_para]:
                    for dst_sent in para_to_nodes[dst_para]:
                        try:
                            cost = nx.shortest_path_length(
                                g, src_sent, dst_sent, weight="cost",
                            )
                            if cost < best_cost:
                                best_cost = cost
                                best_sent_path = nx.shortest_path(
                                    g, src_sent, dst_sent, weight="cost",
                                )
                        except nx.NetworkXNoPath:
                            continue

        # Fallback for disconnected Q-relevant paragraphs:
        # If Q-relevant paragraphs exist but no path was found between any pair
        # (they're in different connected components), use them directly as the
        # chain — they're still the most question-relevant paragraphs.
        if best_sent_path is None:
            # Create an artificial path using the first sentence of each
            # Q-relevant paragraph
            artificial_nodes = []
            for p_idx in q_relevant:
                if para_to_nodes[p_idx]:
                    artificial_nodes.append(para_to_nodes[p_idx][0])
            if len(artificial_nodes) >= 2:
                best_sent_path = artificial_nodes

    elif len(q_relevant) == 1:
        # Only one Q-relevant paragraph: expand via strongest cross-paragraph neighbor
        src_para = q_relevant[0]
        best_strength = -1.0
        for src_sent in para_to_nodes[src_para]:
            for nb in g.neighbors(src_sent):
                nb_para = g.nodes[nb]["para_idx"]
                if nb_para == src_para:
                    continue  # skip same-paragraph neighbors
                strength = g[src_sent][nb].get("strength", 0)
                if strength > best_strength:
                    best_strength = strength
                    best_sent_path = [src_sent, nb]

        # If no graph neighbors, use paragraph-level entity overlap fallback
        if best_sent_path is None:
            q_ents = extract_entities(question)
            best_overlap = 0
            best_para = -1
            for p_idx in range(len(titles)):
                if p_idx == src_para:
                    continue
                full_text = titles[p_idx] + " " + " ".join(sentences_list[p_idx])
                p_ents = extract_entities(full_text)
                overlap_count = len(p_ents & q_ents) + len(
                    p_ents & para_entities[src_para]
                )
                if overlap_count > best_overlap:
                    best_overlap = overlap_count
                    best_para = p_idx
            if best_para >= 0 and para_to_nodes[src_para] and para_to_nodes[best_para]:
                best_sent_path = [para_to_nodes[src_para][0],
                                  para_to_nodes[best_para][0]]

    # --- Map sentence path to paragraph chain ---
    if best_sent_path is not None and len(best_sent_path) >= 2:
        # Extract unique paragraph indices in path order
        para_chain: list[int] = []
        for node in best_sent_path:
            p_idx = g.nodes[node]["para_idx"]
            if not para_chain or para_chain[-1] != p_idx:
                para_chain.append(p_idx)

        if len(para_chain) >= 2:
            chain = para_chain
            chain_found = True

            # Extract bridge entities from cross-paragraph transitions
            for k in range(len(best_sent_path) - 1):
                u, v = best_sent_path[k], best_sent_path[k + 1]
                p_u = g.nodes[u]["para_idx"]
                p_v = g.nodes[v]["para_idx"]
                if p_u == p_v:
                    continue  # skip intra-paragraph edges

                edge_data = g.get_edge_data(u, v) or {}
                edge_type = edge_data.get("edge_type", "unknown")
                ents = edge_data.get("entities", set())
                via = sorted(ents) if ents else []

                if not via and edge_data.get("title_ref"):
                    via = [f"(title-ref: {titles[p_v]})"]
                if not via and edge_type == "similarity":
                    cs = edge_data.get("cos_sim", 0)
                    via = [f"(content similarity: {cs:.2f})"]

                if via:
                    bridge_entities.append({
                        "from": titles[p_u],
                        "to": titles[p_v],
                        "via": via,
                    })

            # Deduplicate bridge entities (same from→to pair)
            seen: set[tuple[str, str]] = set()
            deduped: list[dict[str, Any]] = []
            for be in bridge_entities:
                key = (be["from"], be["to"])
                if key not in seen:
                    seen.add(key)
                    deduped.append(be)
            bridge_entities = deduped

    # Fallback: centrality-based (sentence-level → paragraph mapping)
    if not chain_found and g.number_of_edges() > 0:
        centrality = nx.degree_centrality(g)
        # Group centrality by paragraph
        para_centrality: dict[int, float] = defaultdict(float)
        for node, cent in centrality.items():
            para_centrality[g.nodes[node]["para_idx"]] += cent

        top_paras = sorted(para_centrality, key=para_centrality.get,
                           reverse=True)[:3]
        if len(top_paras) >= 2:
            # Try shortest path between top-centrality paragraphs
            best_cost_fb = float("inf")
            for src_sent in para_to_nodes[top_paras[0]]:
                for dst_sent in para_to_nodes[top_paras[1]]:
                    try:
                        cost = nx.shortest_path_length(
                            g, src_sent, dst_sent, weight="cost",
                        )
                        if cost < best_cost_fb:
                            best_cost_fb = cost
                            fb_path = nx.shortest_path(
                                g, src_sent, dst_sent, weight="cost",
                            )
                            # Map to paragraph chain
                            pc: list[int] = []
                            for node in fb_path:
                                p = g.nodes[node]["para_idx"]
                                if not pc or pc[-1] != p:
                                    pc.append(p)
                            if len(pc) >= 2:
                                chain = pc
                                chain_found = True
                    except nx.NetworkXNoPath:
                        continue

    # --- Expand chain to improve recall ---
    # After finding the core chain (shortest path), add paragraphs
    # that are strongly connected to both the question and chain paragraphs.
    if chain_found and len(chain) >= 2:
        chain = _expand_chain(
            chain, question, titles, sentences_list, para_entities, max_extra=3,
        )

    return {
        "chain": chain,
        "bridge_entities": bridge_entities,
        "topology": topology,
        "chain_found": chain_found,
        "q_relevant": q_relevant,
    }


# ---------------------------------------------------------------------------
# Chain expansion: add high-overlap paragraphs to improve recall
# ---------------------------------------------------------------------------

def _expand_chain(
    chain: list[int],
    question: str,
    titles: list[str],
    sentences_list: list[list[str]],
    para_entities: list[set[str]],
    max_extra: int = 3,
    min_score: float = 2.0,
) -> list[int]:
    """Expand the chain by adding paragraphs with high entity overlap.

    Scoring: For each non-chain paragraph, compute
        score = |para_ents ∩ q_ents| + |para_ents ∩ chain_ents|
                + 2.0 * (title mentioned in chain text)
    Add up to ``max_extra`` paragraphs that score >= ``min_score``.
    New paragraphs are inserted at the position where they have the
    strongest connection (entity overlap) with existing chain members.
    """
    chain_set = set(chain)
    q_ents = extract_entities(question)

    # Aggregate entities from all chain paragraphs
    chain_ents: set[str] = set()
    chain_text_lower = ""
    for p_idx in chain:
        chain_ents |= para_entities[p_idx]
        chain_text_lower += " " + titles[p_idx].lower() + " " + " ".join(
            sentences_list[p_idx]
        ).lower()

    # Score candidate paragraphs
    candidates: list[tuple[float, int]] = []
    for p_idx in range(len(titles)):
        if p_idx in chain_set:
            continue
        p_ents = para_entities[p_idx]
        q_overlap = len(p_ents & q_ents)
        chain_overlap = len(p_ents & chain_ents)
        # Bonus: paragraph's title mentioned in chain text
        title_bonus = 2.0 if titles[p_idx].lower() in chain_text_lower else 0.0
        # Bonus: chain paragraph title mentioned in this paragraph's text
        p_text_lower = titles[p_idx].lower() + " " + " ".join(
            sentences_list[p_idx]
        ).lower()
        reverse_title_bonus = 0.0
        for c_idx in chain:
            if titles[c_idx].lower() in p_text_lower:
                reverse_title_bonus = 2.0
                break

        score = q_overlap + chain_overlap + title_bonus + reverse_title_bonus
        if score >= min_score:
            candidates.append((score, p_idx))

    # Sort by score descending, take top max_extra
    candidates.sort(key=lambda x: -x[0])
    extras = [p_idx for _, p_idx in candidates[:max_extra]]

    if not extras:
        return chain

    # Insert extras at the best position in the chain
    # Simple heuristic: append after the chain member with highest overlap
    expanded = list(chain)
    for extra_idx in extras:
        extra_ents = para_entities[extra_idx]
        best_pos = len(expanded)  # default: append at end
        best_overlap = -1
        for pos, c_idx in enumerate(expanded):
            overlap = len(extra_ents & para_entities[c_idx])
            if overlap > best_overlap:
                best_overlap = overlap
                best_pos = pos + 1  # insert after this chain member
        expanded.insert(best_pos, extra_idx)

    return expanded


# ---------------------------------------------------------------------------
# Reasoning guide formatting (paragraph-level — unchanged from v10b)
# ---------------------------------------------------------------------------

def format_reasoning_guide(chain_info: dict[str, Any]) -> str:
    """Format the reasoning chain into a structured prompt guide.

    Produces output like:
        REASONING CHAIN (follow this path):
          Step 1: "[Title A]" connects to "[Title B]" via [Ted Mosby]
        KEY PARAGRAPHS (in reasoning order):
          1. [Title A]
          2. [Title B]
        BRIDGE ENTITIES: Ted Mosby
    """
    if not chain_info["chain_found"]:
        return ("No clear reasoning chain detected. "
                "Read all paragraphs carefully and trace entity connections.")

    lines: list[str] = []
    bridge_ents = chain_info["bridge_entities"]

    # Reasoning chain steps
    if bridge_ents:
        lines.append("REASONING CHAIN (follow this path):")
        for step_i, be in enumerate(bridge_ents, 1):
            via_str = ", ".join(be["via"][:3])  # cap at 3 entities per step
            lines.append(
                f'  Step {step_i}: "[{be["from"]}]" connects to '
                f'"[{be["to"]}]" via [{via_str}]'
            )
        lines.append("")

    # Key paragraphs in reasoning order
    if bridge_ents:
        ordered_titles: list[str] = [bridge_ents[0]["from"]]
        for be in bridge_ents:
            if be["to"] not in ordered_titles:
                ordered_titles.append(be["to"])

        lines.append("KEY PARAGRAPHS (in reasoning order):")
        for i, t in enumerate(ordered_titles, 1):
            lines.append(f"  {i}. [{t}]")
        lines.append("")

    # All bridge entities
    all_bridge: set[str] = set()
    for be in bridge_ents:
        for v in be["via"]:
            if not v.startswith("("):
                all_bridge.add(v)
    if all_bridge:
        lines.append(f"BRIDGE ENTITIES: {', '.join(sorted(all_bridge))}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: reorder paragraphs with chain first (unchanged from v10b)
# ---------------------------------------------------------------------------

def reorder_paragraphs(
    titles: list[str],
    sentences_list: list[list[str]],
    chain: list[int],
) -> tuple[list[str], list[list[str]]]:
    """Reorder paragraphs so chain paragraphs come first.

    Returns new (titles, sentences_list) with chain paragraphs at the front
    and remaining paragraphs in their original order after.
    """
    n = len(titles)
    chain_set = set(chain)

    # Chain paragraphs first (in chain order)
    new_titles: list[str] = [titles[i] for i in chain]
    new_sents: list[list[str]] = [sentences_list[i] for i in chain]

    # Remaining paragraphs in original order
    for i in range(n):
        if i not in chain_set:
            new_titles.append(titles[i])
            new_sents.append(sentences_list[i])

    return new_titles, new_sents
