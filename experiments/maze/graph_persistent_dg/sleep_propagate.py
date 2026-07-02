"""Sleep phase: reward propagation on knowledge graph.

Propagates node-level rewards through graph edges (Q-learning style)
so that each node accumulates information about what lies ahead.

Usage:
    optimized = sleep_optimize(wake1_graph, gamma=0.95, n_iters=50)
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np


def propagate_rewards(
    graph: nx.Graph,
    gamma: float = 0.95,
    n_iters: int = 50,
) -> None:
    """Propagate node rewards through edges (in-place).

    For each node n:
        propagated(n) = reward(n) + gamma * max(propagated(neighbor))

    Positive rewards (goal) propagate backward along the path,
    making upstream nodes attractive. Negative rewards (dead-end)
    propagate similarly, making upstream nodes unattractive.

    Args:
        graph: nx.Graph with node attribute ``reward`` (float).
        gamma: Discount factor controlling propagation reach.
        n_iters: Maximum iterations (stops early on convergence).
    """
    # Initialise: propagated = own reward
    for _node, data in graph.nodes(data=True):
        data["propagated"] = data.get("reward", 0.0)

    for _ in range(n_iters):
        updated = False
        for node, data in graph.nodes(data=True):
            neighbors_prop = [
                graph.nodes[nb].get("propagated", 0.0)
                for nb in graph.neighbors(node)
            ]
            best_neighbor = max(neighbors_prop, default=0.0)
            new_val = data.get("reward", 0.0) + gamma * best_neighbor

            if abs(new_val - data["propagated"]) > 1e-6:
                data["propagated"] = new_val
                updated = True

        if not updated:
            break  # converged


def sleep_optimize(
    graph: nx.Graph,
    gamma: float = 0.95,
    n_iters: int = 50,
    prune: bool = False,
    prune_threshold: float = -0.8,
) -> nx.Graph:
    """Run full sleep phase: propagate rewards and optionally prune.

    Args:
        graph: Wake1 graph with node ``reward`` attributes.
        gamma: Discount factor for propagation.
        n_iters: Max propagation iterations.
        prune: If True, remove edges to nodes with very low propagated values.
        prune_threshold: Threshold below which nodes are pruned.

    Returns:
        Optimised copy of the graph ready for Wake2.
    """
    optimized = graph.copy()

    # 1. Reward propagation
    propagate_rewards(optimized, gamma=gamma, n_iters=n_iters)

    # 2. Remove isolated nodes
    optimized.remove_nodes_from(list(nx.isolates(optimized)))

    # 3. Optional: prune nodes with very negative propagated values
    if prune:
        to_remove = [
            n for n, d in optimized.nodes(data=True)
            if d.get("propagated", 0.0) < prune_threshold
        ]
        optimized.remove_nodes_from(to_remove)
        # Clean up newly isolated nodes
        optimized.remove_nodes_from(list(nx.isolates(optimized)))

    # 4. Sync propagated values into abs_vector dim9 (for extended vector mode)
    sync_vectors(optimized)

    # 5. Record propagation strength as edge weights
    for u, v in optimized.edges():
        try:
            pu = float(optimized.nodes[u].get("propagated", 0.0))
            pv = float(optimized.nodes[v].get("propagated", 0.0))
            optimized[u][v]["propagation_weight"] = max(pu, pv)
        except Exception:
            pass

    return optimized


def sleep_replay_optimize(
    graph: nx.Graph,
    sleep_q: dict | None,
) -> nx.Graph:
    """Sleep variant 'replay': write trajectory-based Q values onto the graph.

    Undirected max-propagation (sleep_optimize) self-reinforces to
    ~reward/(1-gamma), inflating every node positive and saturating
    tanh-dim9 (see test/test_sleep_propagate_semantics.py). This variant
    instead takes the Q(s, a) table built by qhlib.sleep.build_sleep_q_table
    — a DIRECTED episodic backup over experienced transitions, with absorbing
    goal and negative-example penalties — and stores it as node 'propagated':

        direction node (r, c, a)  -> Q((r, c), a)
        query node     (r, c, -1) -> max_a Q((r, c), a)   (state value)

    Q values are bounded (goal_reward=1.0 scale), so tanh does not saturate,
    and negative examples survive because backups follow the trajectory
    instead of an undirected max over neighbors.

    Cleanup and dim8/dim9 sync are identical to sleep_optimize.
    """
    optimized = graph.copy()
    table = sleep_q or {}

    for node, data in optimized.nodes(data=True):
        try:
            r, c, d = int(node[0]), int(node[1]), int(node[2])
        except Exception:
            data["propagated"] = 0.0
            continue
        qs = table.get((r, c), {}) or {}
        if d >= 0:  # direction node: Q(s, a)
            data["propagated"] = float(qs.get(d, qs.get(str(d), 0.0)))
        else:  # query node: V(s) = max_a Q(s, a)
            data["propagated"] = float(max(qs.values())) if qs else 0.0

    # Same cleanup as sleep_optimize step 2
    optimized.remove_nodes_from(list(nx.isolates(optimized)))

    # Same vector sync as sleep_optimize step 4
    sync_vectors(optimized)

    # Same edge annotation as sleep_optimize step 5
    for u, v in optimized.edges():
        try:
            pu = float(optimized.nodes[u].get("propagated", 0.0))
            pv = float(optimized.nodes[v].get("propagated", 0.0))
            optimized[u][v]["propagation_weight"] = max(pu, pv)
        except Exception:
            pass

    return optimized


def sync_vectors(graph: nx.Graph) -> None:
    """Sync reward/propagated into abs_vector dims 8-9 (in-place).

    Only updates nodes whose abs_vector has >= 10 dimensions (extended mode).
    Uses tanh(propagated) to squash into [-1, 1] for distance computation.
    """
    for _node, data in graph.nodes(data=True):
        vec = data.get("abs_vector")
        if vec is None:
            continue
        arr = np.asarray(vec, dtype=float)
        if arr.size < 10:
            continue
        arr[8] = float(data.get("reward", 0.0))
        arr[9] = math.tanh(float(data.get("propagated", 0.0)))
        data["abs_vector"] = arr
