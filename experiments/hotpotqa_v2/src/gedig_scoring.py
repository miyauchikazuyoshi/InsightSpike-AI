"""Spec H: geDIG-based document scoring for BRIGHT pipeline.

Ports core algorithms from insightspike (MessagePassing, EdgeReevaluator)
as pure numpy/networkx implementations, and adds per-document geDIG scoring.

geDIG formula (per-document, local subgraph):
  geDIG_d = Δ_GED_local - λ · (Δ_H_local + β_sp · Δ_SP_local)

Negative geDIG → high information integration → document is relevant.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# geDIG lightweight functions (ported from gedig_router.py)
# ---------------------------------------------------------------------------

def _local_ged(g_before: nx.Graph, g_after: nx.Graph) -> float:
    """Normalized GED between two local subgraphs.

    GED = (|added_nodes| + |added_edges| + |removed_edges|) / max(|E_before|, 1)
    """
    nodes_before = set(g_before.nodes())
    nodes_after = set(g_after.nodes())
    edges_before = set(g_before.edges())
    edges_after = set(g_after.edges())

    added_nodes = len(nodes_after - nodes_before)
    added_edges = len(edges_after - edges_before)
    removed_edges = len(edges_before - edges_after)

    denominator = max(len(edges_before), 1)
    return (added_nodes + added_edges + removed_edges) / denominator


def _local_entropy(features: np.ndarray, n_bins: int = 32) -> float:
    """Shannon entropy of feature distribution (averaged over dimensions).

    Ported from gedig_router.py _shannon_entropy_features().
    """
    if features.ndim != 2 or features.shape[0] < 2:
        return 0.0

    n_nodes, n_dim = features.shape

    # Subsample dimensions for efficiency
    rng = np.random.RandomState(42)
    if n_dim > 64:
        dim_idx = rng.choice(n_dim, 64, replace=False)
        features = features[:, dim_idx]
        n_dim = 64

    total_entropy = 0.0
    for d in range(n_dim):
        col = features[:, d]
        col_range = col.max() - col.min()
        if col_range < 1e-10:
            continue
        bins = np.linspace(col.min() - 1e-10, col.max() + 1e-10, n_bins + 1)
        hist, _ = np.histogram(col, bins=bins)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        total_entropy -= np.sum(probs * np.log(probs))

    return total_entropy / max(n_dim, 1)


def _local_sp_gain(g_before: nx.Graph, g_after: nx.Graph,
                   sample_pairs: int = 50) -> float:
    """Relative shortest-path improvement (local subgraph version).

    Ported from gedig_router.py _sp_gain(), with smaller sample count
    for per-document local subgraphs.
    """
    common_nodes = list(set(g_before.nodes()) & set(g_after.nodes()))
    if len(common_nodes) < 2:
        return 0.0

    rng = np.random.RandomState(42)
    n = len(common_nodes)
    n_pairs = min(sample_pairs, n * (n - 1) // 2)

    pairs: set[tuple[int, int]] = set()
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 10:
        i, j = rng.randint(0, n, size=2)
        if i != j:
            pairs.add((min(i, j), max(i, j)))
        attempts += 1

    if not pairs:
        return 0.0

    total_gain = 0.0
    valid_pairs = 0

    for i, j in pairs:
        u, v = common_nodes[i], common_nodes[j]
        try:
            sp_before = nx.shortest_path_length(g_before, u, v)
        except nx.NetworkXNoPath:
            sp_before = None
        try:
            sp_after = nx.shortest_path_length(g_after, u, v)
        except nx.NetworkXNoPath:
            sp_after = None

        if sp_before is not None and sp_after is not None:
            if sp_before > 0:
                gain = (sp_before - sp_after) / sp_before
                total_gain += gain
                valid_pairs += 1
        elif sp_before is None and sp_after is not None:
            total_gain += 1.0
            valid_pairs += 1
        elif sp_before is not None and sp_after is None:
            total_gain -= 1.0
            valid_pairs += 1

    return total_gain / max(valid_pairs, 1)


# ---------------------------------------------------------------------------
# Module 1: MessagePassingNX
# ---------------------------------------------------------------------------

class MessagePassingNX:
    """Query-aware message passing on NetworkX graphs.

    Ported from src/insightspike/graph/message_passing.py.
    Pure numpy/networkx implementation (no PyTorch dependency).

    Propagates query relevance through the graph: nodes close to the query
    influence their neighbors, allowing distant but graph-connected nodes
    to receive query signal.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        iterations: int = 2,
        aggregation: str = "weighted_mean",
        self_loop_weight: float = 0.5,
        decay_factor: float = 0.8,
    ):
        self.alpha = alpha
        self.iterations = iterations
        self.aggregation = aggregation
        self.self_loop_weight = self_loop_weight
        self.decay_factor = decay_factor

    def forward(
        self,
        graph: nx.Graph,
        node_features: np.ndarray,
        query_vector: np.ndarray,
    ) -> np.ndarray:
        """Perform message passing on the graph.

        Args:
            graph: NetworkX graph (entity_graph output).
            node_features: (N, D) node feature matrix.
            query_vector: (D,) query feature vector.

        Returns:
            (N, D) updated node representations after message passing.
        """
        N = node_features.shape[0]
        if N == 0:
            return node_features.copy()

        node_list = sorted(graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(node_list)}

        # Ensure query_vector is 2D for cosine_similarity
        qv = query_vector.reshape(1, -1) if query_vector.ndim == 1 else query_vector

        # Dimension alignment
        D_feat = node_features.shape[1]
        D_query = qv.shape[1]
        if D_feat != D_query:
            min_d = min(D_feat, D_query)
            node_features = node_features[:, :min_d]
            qv = qv[:, :min_d]

        # Initial query relevance per node
        norms_feat = np.linalg.norm(node_features, axis=1, keepdims=True)
        norms_feat = np.where(norms_feat < 1e-10, 1.0, norms_feat)
        relevance_scores = cosine_similarity(
            node_features / norms_feat, qv
        ).flatten()
        relevance_scores = np.clip(relevance_scores, 0.0, 1.0)

        # Initialize representations with query influence
        qv_flat = qv.flatten()
        h = np.zeros_like(node_features)
        for i in range(N):
            rel = relevance_scores[i]
            base = (1.0 - self.alpha * rel) * node_features[i]
            inject = (self.alpha * rel) * qv_flat
            h[i] = base + inject

        # Build adjacency from graph edges
        adjacency: dict[int, list[int]] = {i: [] for i in range(N)}
        for u, v in graph.edges():
            ui = node_to_idx.get(u)
            vi = node_to_idx.get(v)
            if ui is not None and vi is not None:
                adjacency[ui].append(vi)
                adjacency[vi].append(ui)

        # Message passing iterations
        for _t in range(self.iterations):
            h_new = np.zeros_like(h)

            for node_idx in range(N):
                neighbors = adjacency[node_idx]
                if not neighbors:
                    h_new[node_idx] = h[node_idx]
                    continue

                # Collect neighbor messages
                neighbor_vecs = []
                weights = []
                for nb_idx in neighbors:
                    # Weight: similarity * (1 + neighbor's query relevance * decay)
                    sim = cosine_similarity(
                        h[node_idx].reshape(1, -1),
                        h[nb_idx].reshape(1, -1),
                    )[0, 0]
                    w = sim * (1.0 + relevance_scores[nb_idx] * self.decay_factor)
                    neighbor_vecs.append(h[nb_idx])
                    weights.append(w)

                # Aggregate
                if self.aggregation == "weighted_mean":
                    w_arr = np.array(weights, dtype=np.float64)
                    total = w_arr.sum()
                    if total <= 1e-12:
                        aggregated = np.mean(neighbor_vecs, axis=0)
                    else:
                        w_arr = w_arr / total
                        aggregated = np.average(neighbor_vecs, axis=0, weights=w_arr)
                elif self.aggregation == "max":
                    aggregated = np.max(neighbor_vecs, axis=0)
                else:
                    aggregated = np.mean(neighbor_vecs, axis=0)

                # Self-loop update
                h_new[node_idx] = (
                    self.self_loop_weight * h[node_idx]
                    + (1.0 - self.self_loop_weight) * aggregated
                )

            h = h_new

        return h


# ---------------------------------------------------------------------------
# Module 2: EdgeReevaluatorNX
# ---------------------------------------------------------------------------

class EdgeReevaluatorNX:
    """Dynamic edge discovery/removal based on updated representations.

    Ported from src/insightspike/graph/edge_reevaluator.py.
    Pure numpy/networkx implementation.

    After message passing, node representations have changed. This module:
    1. Removes edges between nodes that are no longer similar.
    2. Discovers new edges between nodes that became similar after propagation.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        new_edge_threshold: float = 0.8,
        max_new_edges_per_node: int = 5,
        edge_decay_factor: float = 0.9,
    ):
        self.similarity_threshold = similarity_threshold
        self.new_edge_threshold = new_edge_threshold
        self.max_new_edges_per_node = max_new_edges_per_node
        self.edge_decay_factor = edge_decay_factor

    def reevaluate(
        self,
        graph: nx.Graph,
        updated_features: np.ndarray,
        query_vector: np.ndarray,
    ) -> tuple[nx.Graph, int, int]:
        """Reevaluate edges based on updated node representations.

        Args:
            graph: Original entity graph (not modified).
            updated_features: (N, D) from MessagePassingNX.
            query_vector: (D,) query vector.

        Returns:
            (refined_graph, n_edges_discovered, n_edges_removed)
        """
        N = updated_features.shape[0]
        if N < 2:
            return graph.copy(), 0, 0

        node_list = sorted(graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(node_list)}

        # Query relevance
        qv = query_vector.reshape(1, -1) if query_vector.ndim == 1 else query_vector
        D_feat = updated_features.shape[1]
        D_query = qv.shape[1]
        if D_feat != D_query:
            min_d = min(D_feat, D_query)
            features = updated_features[:, :min_d]
            qv = qv[:, :min_d]
        else:
            features = updated_features

        query_relevance = cosine_similarity(features, qv).flatten()
        query_relevance = np.clip(query_relevance, 0.0, 1.0)

        # Create refined graph (copy nodes and attributes, rebuild edges)
        refined = nx.Graph()
        for n in node_list:
            refined.add_node(n, **graph.nodes[n])

        n_removed = 0
        n_kept = 0

        # Re-evaluate existing edges
        existing_pairs: set[tuple] = set()
        for u, v, data in graph.edges(data=True):
            ui = node_to_idx.get(u)
            vi = node_to_idx.get(v)
            if ui is None or vi is None:
                continue

            pair = (min(ui, vi), max(ui, vi))
            existing_pairs.add(pair)

            # Cosine similarity between updated representations
            sim = cosine_similarity(
                features[ui].reshape(1, -1),
                features[vi].reshape(1, -1),
            )[0, 0]

            relevance_boost = (query_relevance[ui] + query_relevance[vi]) / 2.0
            adjusted_sim = sim * (1.0 + 0.2 * relevance_boost)

            # Retention check
            keep = adjusted_sim >= self.similarity_threshold
            if not keep:
                # High-relevance retention: if either endpoint is very query-relevant
                if max(query_relevance[ui], query_relevance[vi]) >= 0.8:
                    keep = True
                    adjusted_sim = max(adjusted_sim, self.similarity_threshold * 0.85)

            if keep:
                # Blend old weight with new similarity
                old_weight = data.get("weight", adjusted_sim)
                if isinstance(old_weight, (int, float)):
                    new_weight = (
                        self.edge_decay_factor * old_weight
                        + (1.0 - self.edge_decay_factor) * adjusted_sim
                    )
                else:
                    new_weight = adjusted_sim

                edge_data = dict(data)
                edge_data["weight"] = new_weight
                edge_data["strength"] = new_weight
                refined.add_edge(u, v, **edge_data)
                n_kept += 1
            else:
                n_removed += 1

        # Discover new edges
        n_discovered = 0
        if N >= 2:
            sim_matrix = cosine_similarity(features)
            per_node_count = [0] * N

            # Collect candidates
            candidates = []
            for i in range(N):
                for j in range(i + 1, N):
                    if (i, j) in existing_pairs:
                        continue
                    sim = float(sim_matrix[i, j])
                    if sim >= self.new_edge_threshold:
                        boost = (query_relevance[i] + query_relevance[j]) / 2.0
                        adj_sim = sim * (1.0 + 0.3 * boost)
                        candidates.append((i, j, adj_sim))

            # Sort by adjusted similarity, add best edges
            candidates.sort(key=lambda c: c[2], reverse=True)
            for ci, cj, adj_sim in candidates:
                if (per_node_count[ci] < self.max_new_edges_per_node
                        and per_node_count[cj] < self.max_new_edges_per_node):
                    u = node_list[ci]
                    v = node_list[cj]
                    refined.add_edge(
                        u, v,
                        edge_type="gedig_discovered",
                        weight=adj_sim,
                        strength=adj_sim,
                        cost=max(0.1, 1.0 - adj_sim),
                    )
                    per_node_count[ci] += 1
                    per_node_count[cj] += 1
                    n_discovered += 1

        logger.debug(
            "EdgeReevaluatorNX: kept=%d, removed=%d, discovered=%d",
            n_kept, n_removed, n_discovered,
        )
        return refined, n_discovered, n_removed


# ---------------------------------------------------------------------------
# Module 3: GeDIGDocScorer
# ---------------------------------------------------------------------------

class GeDIGDocScorer:
    """Per-document scoring using geDIG principles.

    For each document, extracts local subgraphs (before and after query
    injection + message passing + edge reevaluation), then computes local
    geDIG components as the document's relevance score.

    geDIG_local = Δ_GED - λ · (Δ_H + β · Δ_SP)

    Negative geDIG → high information integration → relevant document.
    """

    def __init__(
        self,
        lambda_weight: float = 1.0,
        sp_beta: float = 0.5,
        k_hop: int = 2,
        query_connect_top_k: int = 10,
    ):
        self.lambda_weight = lambda_weight
        self.sp_beta = sp_beta
        self.k_hop = k_hop
        self.query_connect_top_k = query_connect_top_k

    def inject_query_node(
        self,
        graph: nx.Graph,
        node_features: np.ndarray,
        query_vector: np.ndarray,
        query_entities: set[str],
    ) -> tuple[nx.Graph, np.ndarray, Any]:
        """Add query as a virtual node connected to most relevant nodes.

        Ported from src/insightspike/graph/construction.py add_query_node().

        Returns:
            (graph_with_query, features_with_query, query_node_id)
        """
        g = graph.copy()
        node_list = sorted(graph.nodes())
        N = len(node_list)

        # Choose unique query node ID
        if node_list and isinstance(node_list[-1], int):
            query_node_id = max(node_list) + 1
        else:
            query_node_id = "QUERY"

        g.add_node(
            query_node_id,
            para_idx=-1,
            sent_idx=-1,
            title="query",
            text="",
            node_type="query",
        )

        # Append query vector to feature matrix
        qv = query_vector.reshape(1, -1) if query_vector.ndim == 1 else query_vector
        D_feat = node_features.shape[1] if node_features.ndim == 2 else 0
        D_query = qv.shape[1]
        if D_feat != D_query and D_feat > 0:
            min_d = min(D_feat, D_query)
            node_features = node_features[:, :min_d]
            qv = qv[:, :min_d]

        features_with_query = np.vstack([node_features, qv])

        # Connect query to most relevant nodes (cosine similarity + entity boost)
        if N == 0:
            return g, features_with_query, query_node_id

        sims = cosine_similarity(node_features, qv).flatten()

        # Entity boost
        from entity_graph import extract_entities
        from bright_cot_pipeline import _extract_lowercase_concepts
        scores = []
        for i, n in enumerate(node_list):
            node_text = g.nodes[n].get("text", "")
            node_ents = extract_entities(node_text)
            node_terms = _extract_lowercase_concepts(node_text)
            all_node_concepts = node_ents | node_terms
            shared = all_node_concepts & query_entities
            concept_boost = len(shared) * 0.1
            combined = float(sims[i]) + concept_boost
            scores.append((n, combined))

        scores.sort(key=lambda x: x[1], reverse=True)

        # Connect to top-K nodes
        for n, score in scores[: self.query_connect_top_k]:
            if score > 0.05:  # Minimum threshold
                g.add_edge(
                    query_node_id, n,
                    edge_type="query_connection",
                    weight=max(0.1, score),
                    strength=max(0.1, score),
                    cost=max(0.05, 1.0 - score),
                )

        return g, features_with_query, query_node_id

    def score_documents(
        self,
        graph_before: nx.Graph,
        graph_after: nx.Graph,
        node_features_before: np.ndarray,
        node_features_after: np.ndarray,
        query_vector: np.ndarray,
        titles: list[str],
        doc_id_map: dict[str, str],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Compute per-document geDIG-based scores.

        For each document:
        1. Collect its nodes in the graph.
        2. Extract k-hop local subgraph (before and after).
        3. Compute local geDIG = Δ_GED - λ·(Δ_H + β·Δ_SP).
        4. Compute message-passing relevance.
        5. Compute shortest-path proximity to query node.
        6. Combined score (negative geDIG = relevant).

        Returns:
            (doc_scores, diagnostics) where doc_scores is {doc_id: score}
            normalized to [0, 1], and diagnostics is {doc_id: {metrics}}.
        """
        node_list_before = sorted(graph_before.nodes())
        node_list_after = sorted(graph_after.nodes())
        node_to_idx_before = {n: i for i, n in enumerate(node_list_before)}
        node_to_idx_after = {n: i for i, n in enumerate(node_list_after)}

        # Query vector for relevance computation
        qv = query_vector.reshape(1, -1) if query_vector.ndim == 1 else query_vector

        # Find query node in after-graph
        query_node = None
        for n in graph_after.nodes():
            if graph_after.nodes[n].get("title") == "query":
                query_node = n
                break

        # Pre-compute shortest paths from query node
        sp_from_query: dict = {}
        if query_node is not None:
            try:
                sp_from_query = dict(
                    nx.single_source_shortest_path_length(
                        graph_after, query_node, cutoff=self.k_hop + 2
                    )
                )
            except Exception:
                sp_from_query = {}

        # --- Pass 1: Collect raw components for all documents ---
        raw_components: list[tuple[str, float, float, float]] = []
        # (doc_id, local_gedig, mp_relevance, sp_proximity)
        skip_docs: set[str] = set()
        diagnostics: dict[str, dict] = {}

        D_feat = node_features_after.shape[1] if node_features_after.ndim == 2 else 0
        D_q = qv.shape[1]
        min_d = min(D_feat, D_q) if D_feat > 0 else 0

        for title in titles:
            doc_id = doc_id_map.get(title, title)

            # Collect nodes belonging to this document
            doc_nodes_before = [
                n for n in node_list_before
                if graph_before.nodes[n].get("title") == title
            ]
            doc_nodes_after = [
                n for n in node_list_after
                if graph_after.nodes[n].get("title") == title
            ]

            if not doc_nodes_before and not doc_nodes_after:
                skip_docs.add(doc_id)
                continue

            # --- Component 1: Local geDIG ---
            local_gedig = self._compute_local_gedig(
                graph_before, graph_after,
                node_features_before, node_features_after,
                doc_nodes_before, doc_nodes_after,
                node_to_idx_before, node_to_idx_after,
            )

            # --- Component 2: Message-passing relevance ---
            mp_relevance = 0.0
            if doc_nodes_after and min_d > 0:
                mp_sims = []
                for n in doc_nodes_after:
                    idx = node_to_idx_after.get(n)
                    if idx is not None and idx < node_features_after.shape[0]:
                        sim = cosine_similarity(
                            node_features_after[idx, :min_d].reshape(1, -1),
                            qv[:, :min_d],
                        )[0, 0]
                        mp_sims.append(float(sim))
                if mp_sims:
                    mp_relevance = np.mean(mp_sims)

            # --- Component 3: Shortest-path proximity to query ---
            sp_proximity = 0.0
            if sp_from_query and doc_nodes_after:
                min_sp = float("inf")
                for n in doc_nodes_after:
                    sp = sp_from_query.get(n)
                    if sp is not None and sp < min_sp:
                        min_sp = sp
                if min_sp < float("inf"):
                    sp_proximity = 1.0 / (1.0 + min_sp)

            raw_components.append((doc_id, local_gedig, mp_relevance, sp_proximity))

        # --- Pass 2: Per-component min-max normalization ---
        doc_scores: dict[str, float] = {}

        if not raw_components:
            for doc_id in skip_docs:
                doc_scores[doc_id] = 0.0
            return doc_scores, diagnostics

        gedig_vals = [-c[1] for c in raw_components]   # negate: higher = better
        mp_vals = [c[2] for c in raw_components]
        sp_vals = [c[3] for c in raw_components]

        def _minmax(vals: list[float]) -> list[float]:
            """Normalize to [0, 1] with min-max scaling."""
            lo, hi = min(vals), max(vals)
            rng = hi - lo
            if rng < 1e-10:
                return [0.5] * len(vals)
            return [(v - lo) / rng for v in vals]

        gedig_norm = _minmax(gedig_vals)
        mp_norm = _minmax(mp_vals)
        sp_norm = _minmax(sp_vals)

        # Component weights (tunable)
        w_gedig = 0.35
        w_mp = 0.40
        w_sp = 0.25

        for i, (doc_id, local_gedig, mp_relevance, sp_proximity) in enumerate(raw_components):
            score = (
                w_gedig * gedig_norm[i]
                + w_mp * mp_norm[i]
                + w_sp * sp_norm[i]
            )
            doc_scores[doc_id] = score
            diagnostics[doc_id] = {
                "local_gedig": round(local_gedig, 4),
                "gedig_norm": round(gedig_norm[i], 4),
                "mp_relevance": round(mp_relevance, 4),
                "mp_norm": round(mp_norm[i], 4),
                "sp_proximity": round(sp_proximity, 4),
                "sp_norm": round(sp_norm[i], 4),
                "raw_score": round(score, 4),
            }

        # Skipped docs get score 0
        for doc_id in skip_docs:
            doc_scores[doc_id] = 0.0

        # Final normalization to [0, 1]
        if doc_scores:
            max_s = max(doc_scores.values())
            min_s = min(doc_scores.values())
            rng = max_s - min_s
            if rng > 1e-10:
                doc_scores = {k: (v - min_s) / rng for k, v in doc_scores.items()}
            elif max_s > 0:
                doc_scores = {k: v / max_s for k, v in doc_scores.items()}

        return doc_scores, diagnostics

    def _compute_local_gedig(
        self,
        g_before: nx.Graph,
        g_after: nx.Graph,
        feat_before: np.ndarray,
        feat_after: np.ndarray,
        doc_nodes_before: list,
        doc_nodes_after: list,
        idx_map_before: dict,
        idx_map_after: dict,
    ) -> float:
        """Compute local geDIG around a document's nodes.

        geDIG_local = Δ_GED - λ · (Δ_H + β · Δ_SP)
        """
        # Extract k-hop local subgraphs
        seed_before = set(doc_nodes_before)
        seed_after = set(doc_nodes_after)

        if not seed_before and not seed_after:
            return 0.0

        local_before = self._extract_k_hop(g_before, seed_before, self.k_hop)
        local_after = self._extract_k_hop(g_after, seed_after, self.k_hop)

        if local_before.number_of_nodes() < 2 and local_after.number_of_nodes() < 2:
            return 0.0

        # GED component
        ged = _local_ged(local_before, local_after)

        # IG component (entropy change)
        feat_before_local = self._gather_features(
            local_before, feat_before, idx_map_before
        )
        feat_after_local = self._gather_features(
            local_after, feat_after, idx_map_after
        )
        h_before = _local_entropy(feat_before_local)
        h_after = _local_entropy(feat_after_local)
        delta_h = (h_after - h_before) / max(h_before, 0.01)

        # SP component
        sp = _local_sp_gain(local_before, local_after, sample_pairs=30)

        # geDIG formula
        ig_combined = delta_h + self.sp_beta * sp
        gedig_value = ged - self.lambda_weight * ig_combined

        return gedig_value

    @staticmethod
    def _extract_k_hop(graph: nx.Graph, seed_nodes: set, k: int) -> nx.Graph:
        """Extract k-hop subgraph around seed nodes."""
        if not seed_nodes:
            return nx.Graph()

        # BFS to find all nodes within k hops
        visited: set = set()
        frontier = seed_nodes & set(graph.nodes())
        visited.update(frontier)

        for _ in range(k):
            next_frontier: set = set()
            for n in frontier:
                for nb in graph.neighbors(n):
                    if nb not in visited:
                        next_frontier.add(nb)
                        visited.add(nb)
            frontier = next_frontier
            if not frontier:
                break

        return graph.subgraph(visited).copy()

    @staticmethod
    def _gather_features(
        subgraph: nx.Graph,
        full_features: np.ndarray,
        idx_map: dict,
    ) -> np.ndarray:
        """Gather feature vectors for nodes in a subgraph."""
        if subgraph.number_of_nodes() == 0:
            return np.empty((0, 0), dtype=np.float32)

        vecs = []
        for n in sorted(subgraph.nodes()):
            idx = idx_map.get(n)
            if idx is not None and idx < full_features.shape[0]:
                vecs.append(full_features[idx])

        if not vecs:
            return np.empty((0, 0), dtype=np.float32)

        return np.vstack(vecs)
