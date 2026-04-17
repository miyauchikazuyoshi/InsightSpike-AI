"""
Utilities for geDIG metrics used in the paper/experiments.

This module provides reference implementations for:
  - Entropy estimation with Dirichlet (Jeffreys) smoothing
  - Normalized entropy change ΔH_norm
  - Approximate ASPL via sampled BFS and ΔSP_rel
  - Information gain term ΔIG_norm = ΔH_norm + γ ΔSP_rel

These are lightweight and dependency-free to ease reproduction of the maze/RAG PoC.
"""
from __future__ import annotations

from collections import deque
from math import log
from random import sample
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


def entropy_from_counts(counts: Sequence[int], alpha: float = 0.5) -> float:
    """Shannon entropy with Dirichlet smoothing.

    Args:
        counts: Non-negative integer counts per category.
        alpha: Pseudocount for Dirichlet smoothing (Jeffreys: 0.5).

    Returns:
        Entropy H(p) in nats.
    """
    K = len(counts)
    total = float(sum(counts))
    denom = total + K * alpha
    if denom <= 0:
        return 0.0
    H = 0.0
    for n in counts:
        p = (n + alpha) / denom
        if p > 0:
            H -= p * log(p)
    return H


def delta_H_norm_from_counts(
    counts_before: Sequence[int],
    counts_after: Sequence[int],
    K: int,
    alpha: float = 0.5,
) -> float:
    """Compute ΔH_norm = (H_after - H_before) / log K.

    Args:
        counts_before: Counts within the window before integration.
        counts_after: Counts within the window after integration.
        K: Number of categories.
        alpha: Dirichlet smoothing pseudocount.

    Returns:
        Normalized entropy change ΔH_norm (can be negative when entropy decreases).
    """
    if K <= 1:
        return 0.0
    Hb = entropy_from_counts(counts_before, alpha=alpha)
    Ha = entropy_from_counts(counts_after, alpha=alpha)
    return (Ha - Hb) / max(log(K), 1e-12)


def _bfs_distances(adj: Mapping[int, Sequence[int]], start: int, max_depth: int | None) -> Dict[int, int]:
    seen: Dict[int, int] = {start: 0}
    dq: deque[Tuple[int, int]] = deque([(start, 0)])
    while dq:
        u, d = dq.popleft()
        if max_depth is not None and d >= max_depth:
            continue
        for v in adj.get(u, ()):  # type: ignore[arg-type]
            if v not in seen:
                seen[v] = d + 1
                dq.append((v, d + 1))
    return seen


def aspl_sampled(
    adj: Mapping[int, Sequence[int]],
    nodes: Sequence[int] | None = None,
    M: int = 64,
    max_depth: int | None = None,
) -> float:
    """Estimate ASPL using BFS distances on M sampled node pairs.

    Args:
        adj: Adjacency mapping {node: neighbors} of an undirected simple graph.
        nodes: Optional list of nodes to sample from; defaults to keys of `adj`.
        M: Number of pairs to sample.
        max_depth: Optional BFS depth cutoff (e.g., the current hop horizon).

    Returns:
        Estimated average shortest path length over reachable pairs.
        Returns 0.0 if insufficient pairs are reachable.
    """
    if nodes is None:
        nodes = list(adj.keys())
    if len(nodes) < 2:
        return 0.0
    # Sample pairs without replacement as much as possible
    # Fallback to smaller M if graph is tiny
    m = min(M, len(nodes) * (len(nodes) - 1) // 2)
    if m <= 0:
        return 0.0

    # Build a pool of pairs by random sampling of endpoints, then deduplicate
    pairs: set[Tuple[int, int]] = set()
    tries = 0
    while len(pairs) < m and tries < m * 10:
        u, v = sample(nodes, 2)
        if u != v:
            a, b = (u, v) if u < v else (v, u)
            pairs.add((a, b))
        tries += 1
    if not pairs:
        return 0.0

    total = 0.0
    cnt = 0
    cache: Dict[int, Dict[int, int]] = {}
    for a, b in pairs:
        da = cache.get(a)
        if da is None:
            da = _bfs_distances(adj, a, max_depth)
            cache[a] = da
        d = da.get(b)
        if d is None:
            # Optionally try BFS from b to increase hit rate
            db = cache.get(b)
            if db is None:
                db = _bfs_distances(adj, b, max_depth)
                cache[b] = db
            d = db.get(a)
        if d is not None and d > 0:
            total += float(d)
            cnt += 1
    return total / cnt if cnt > 0 else 0.0


def delta_sp_rel(
    L_before: float,
    L_after: float,
    eps: float = 1e-9,
) -> float:
    """Compute ΔSP_rel = (L_before - L_after) / max(L_before, eps)."""
    if L_before <= eps:
        return 0.0
    return (L_before - L_after) / max(L_before, eps)


def ig_norm(
    dH_norm: float,
    dSP_rel: float,
    gamma: float = 0.2,
) -> float:
    """Compute ΔIG_norm = ΔH_norm + γ ΔSP_rel."""
    return dH_norm + gamma * dSP_rel


__all__ = [
    "entropy_from_counts",
    "delta_H_norm_from_counts",
    "aspl_sampled",
    "delta_sp_rel",
    "ig_norm",
]

