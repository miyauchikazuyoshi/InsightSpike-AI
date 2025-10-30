from __future__ import annotations

from typing import Any, Dict, Optional, Sequence
import math

import networkx as nx
import numpy as np


def normalized_ged(
    g1: nx.Graph,
    g2: nx.Graph,
    *,
    node_cost: float = 1.0,
    edge_cost: float = 1.0,
    normalization: str = "sum",
    efficiency_weight: float = 0.3,
    enable_spectral: bool = False,
    spectral_weight: float = 0.3,
    norm_override: float | None = None,
) -> Dict[str, float]:
    """Compute normalized GED and structural improvement.

    Returns a dict with keys: raw_ged, normalized_ged, structural_improvement, efficiency_change.
    Mirrors the logic used by GeDIGCore._calculate_normalized_ged.
    """
    n1, n2 = g1.number_of_nodes(), g2.number_of_nodes()
    e1, e2 = g1.number_of_edges(), g2.number_of_edges()
    node_ops = abs(n2 - n1) * node_cost
    common_nodes = set(g1.nodes()) & set(g2.nodes())
    common_edges = sum(1 for u, v in g1.edges() if u in common_nodes and v in common_nodes and g2.has_edge(u, v))
    edge_ops = ((e1 - common_edges) + (e2 - common_edges)) * edge_cost
    raw_ged = node_ops + edge_ops

    if norm_override is not None and norm_override > 0:
        norm_factor = float(norm_override)
    elif normalization == "sum":
        norm_factor = (n1 + n2 + e1 + e2) * max(node_cost, edge_cost)
    elif normalization == "max":
        norm_factor = max(n1, n2) * node_cost + max(e1, e2) * edge_cost
    else:  # mean
        norm_factor = ((n1 + n2) / 2 * node_cost + (e1 + e2) / 2 * edge_cost)
    norm_ged = raw_ged / norm_factor if norm_factor > 0 else 0.0

    # Efficiency component (optional blending)
    eff1 = _graph_efficiency(g1)
    eff2 = _graph_efficiency(g2)
    efficiency_change = eff2 - eff1

    base_improvement = -norm_ged
    structural_improvement = base_improvement * (1 - efficiency_weight) + efficiency_change * efficiency_weight
    if structural_improvement <= 0 and efficiency_change > 0:
        structural_improvement = efficiency_change

    if enable_spectral:
        spectral_before = _spectral_score(g1)
        spectral_after = _spectral_score(g2)
        spectral_improvement = (spectral_before - spectral_after) / (spectral_before + 1e-10)
        structural_improvement = structural_improvement * (1 - spectral_weight) + np.tanh(spectral_improvement) * spectral_weight

    structural_improvement = float(np.clip(structural_improvement, -1.0, 1.0))
    structural_cost = float(max(0.0, -structural_improvement))

    return {
        "raw_ged": float(raw_ged),
        "normalized_ged": float(norm_ged),
        "structural_cost": structural_cost,
        "structural_improvement": structural_improvement,
        "efficiency_change": float(efficiency_change),
        "normalization_den": float(norm_factor),
    }


def entropy_ig(
    graph: nx.Graph,
    features_before: np.ndarray,
    features_after: np.ndarray,
    *,
    smoothing: float = 1e-10,
    min_nodes: int = 2,
    epsilon: float = 1e-3,
    norm_strategy: str = "before",
    extra_vectors: Optional[Sequence[Sequence[float]]] = None,
    fixed_den: Optional[float] = None,
    k_star: Optional[int] = None,
    min_candidates: int = 2,
    delta_mode: str = "after_before",
) -> Dict[str, float]:
    """Compute IG as reduction in mean local entropy, normalised by log(candidate_count)."""
    def _ensure_2d(array: np.ndarray) -> np.ndarray:
        if array.ndim == 1:
            return array.reshape(1, -1)
        if array.ndim == 0:
            return array.reshape(1, 1)
        return array

    if graph.number_of_nodes() < min_nodes:
        return {
            "ig_value": 0.0,
            "variance_reduction": 0.0,
            "entropy_before": 0.0,
            "entropy_after": 0.0,
            "delta_entropy": 0.0,
        }

    features_before_arr = np.asarray(features_before, dtype=np.float32)
    features_after_arr = np.asarray(features_after, dtype=np.float32)
    features_before_arr = _ensure_2d(features_before_arr) if features_before_arr.size else np.zeros((0, 0), dtype=np.float32)
    features_after_arr = _ensure_2d(features_after_arr) if features_after_arr.size else np.zeros((0, 0), dtype=np.float32)

    extra_array: Optional[np.ndarray] = None
    if extra_vectors is not None:
        extra_array = np.asarray(extra_vectors, dtype=np.float32)
        if extra_array.ndim == 1:
            extra_array = extra_array.reshape(1, -1)
        elif extra_array.ndim == 0:
            extra_array = extra_array.reshape(1, 1)
        if extra_array.size == 0:
            extra_array = None

    entropies_before = _local_entropies(graph, features_before_arr, smoothing)
    entropies_after = _local_entropies(graph, features_after_arr, smoothing, extra_vectors=extra_array)

    mean_before = float(np.mean(entropies_before)) if entropies_before.size > 0 else 0.0
    mean_after = float(np.mean(entropies_after)) if entropies_after.size > 0 else 0.0
    mode = str(delta_mode or "after_before").lower()
    if mode in ("before_after", "reduction", "entropy_reduction"):
        delta_entropy = mean_before - mean_after
    else:
        # default: after - before
        delta_entropy = mean_after - mean_before

    before_count = int(features_before_arr.shape[0])
    after_count = int(features_after_arr.shape[0])
    extra_count = 0 if extra_array is None else int(extra_array.shape[0])
    candidate_total = max(before_count + extra_count, after_count + extra_count, 1)
    strategy = str(norm_strategy or "before").lower()

    if fixed_den is not None and not math.isnan(fixed_den):
        denom = max(float(fixed_den), epsilon)
    elif strategy in {"before", "h_before", "mean_before"}:
        denom = max(mean_before, epsilon)
    elif strategy in {"max", "h_max", "max_h"}:
        denom = max(max(mean_before, mean_after), epsilon)
    elif strategy in {"two_max", "2max", "two_hmax", "double_max"}:
        denom = max(2.0 * max(mean_before, mean_after), epsilon)
    elif strategy in {"two_before", "double_before", "2before"}:
        denom = max(2.0 * mean_before, epsilon)
    elif strategy in {"logn", "log", "log_candidates"}:
        denom = max(math.log(candidate_total + 1.0), epsilon)
    else:
        denom = max(mean_before, epsilon)

    effective_k = k_star if k_star is not None else candidate_total
    if effective_k < max(1, min_candidates):
        return {
            "ig_value": 0.0,
            "variance_reduction": float(delta_entropy),
            "delta_entropy": float(delta_entropy),
            "entropy_before": mean_before,
            "entropy_after": mean_after,
            "normalization_den": float(denom),
            "candidate_count": float(effective_k),
        }

    if denom <= epsilon:
        ig_value = 0.0
    else:
        ig_value = delta_entropy / denom

    return {
        "ig_value": float(ig_value),
        "variance_reduction": float(delta_entropy),
        "delta_entropy": float(delta_entropy),
        "entropy_before": mean_before,
        "entropy_after": mean_after,
        "normalization_den": float(denom),
        "candidate_count": float(candidate_total),
    }


# ----------------- helpers -----------------

def _graph_efficiency(g: nx.Graph) -> float:
    if g.number_of_nodes() == 0:
        return 0.0
    try:
        ge = nx.global_efficiency(g)
    except Exception:
        ge = 0.0
    try:
        cl = nx.average_clustering(g)
    except Exception:
        cl = 0.0
    return 0.7 * ge + 0.3 * cl


def _spectral_score(g: nx.Graph) -> float:
    if g.number_of_nodes() < 2:
        return 0.0
    try:
        L = nx.laplacian_matrix(g).toarray()
        eig = np.linalg.eigvalsh(L)
        return float(np.std(eig))
    except Exception:
        return 0.0


def _local_entropies(
    graph: nx.Graph,
    features: np.ndarray,
    smoothing: float,
    extra_vectors: Optional[np.ndarray] = None,
) -> np.ndarray:
    entropies = []
    node_index = {node: idx for idx, node in enumerate(graph.nodes())}
    for node in graph.nodes():
        local_nodes = [node] + list(graph.neighbors(node))
        local_feats = []
        for n in local_nodes:
            if isinstance(n, (int, np.integer)):
                idx = int(n)
            else:
                idx = node_index.get(n)
            if idx is None:
                continue
            if 0 <= idx < len(features):
                local_feats.append(features[idx])
        if not local_feats:
            if extra_vectors is None or extra_vectors.size == 0:
                continue
            lf = np.zeros((0, extra_vectors.shape[1]), dtype=np.float32)
        else:
            lf = np.array(local_feats, dtype=np.float32)
        if extra_vectors is not None and extra_vectors.size > 0:
            lf = _append_with_alignment(lf, extra_vectors)
        if len(lf) > 1:
            normed = lf / (np.linalg.norm(lf, axis=1, keepdims=True) + smoothing)
            sims = np.dot(normed, normed.T)
            probs = (sims + 1) / 2
            probs = probs.flatten(); probs = probs / (probs.sum() + smoothing)
            entropy = -np.sum(probs * np.log(probs + smoothing))
        else:
            entropy = 0.0
        entropies.append(entropy)
    return np.array(entropies)


def graph_efficiency(g: nx.Graph) -> float:
    return _graph_efficiency(g)


def spectral_score(g: nx.Graph) -> float:
    return _spectral_score(g)


def local_entropies(graph: nx.Graph, features: np.ndarray, smoothing: float = 1e-10) -> np.ndarray:
    return _local_entropies(graph, features, smoothing)


def _append_with_alignment(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0:
        width = b.shape[1]
        return b.copy()
    width = max(a.shape[1], b.shape[1])
    if a.shape[1] != width:
        pad = np.zeros((a.shape[0], width - a.shape[1]), dtype=a.dtype)
        a = np.hstack([a, pad])
    if b.shape[1] != width:
        pad = np.zeros((b.shape[0], width - b.shape[1]), dtype=b.dtype)
        b = np.hstack([b, pad])
    return np.vstack([a, b])
