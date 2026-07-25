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

    # v7: materialise β₁ cycle-size signals on shortcut edges and project
    # them to the source query-state/action pairs used during Wake2.
    annotate_dg_size(optimized)

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

    # v7: annotate β₁ cycle size per shortcut (DG signal for action bias).
    # Computed here at sleep (offline, affordable); read at action selection.
    annotate_dg_size(optimized)

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


def annotate_dg_size(graph: nx.Graph) -> None:
    """Materialise the v7 β₁ cycle-size proxy in-place.

    A shortcut is an abstract graph edge whose endpoint cells are not spatially
    adjacent. Its size is the corridor-only shortest path between those cells
    plus the closing shortcut edge. The value is stored on the shortcut edge.

    Query/direction graphs encode a recall as ``Q(current) -- D(memory, action)``.
    For action selection, the edge value is therefore also projected onto the
    query node as ``dg_action_sizes[action]`` (maximum when several recalled
    direction nodes suggest the same action). ``dg_size`` on direction nodes is
    retained as an endpoint-level diagnostic, but Wake2 readout uses the query
    projection so it evaluates the same shortcut edge that created the signal.

    Manhattan-1 edges are treated as corridor edges by this lightweight proxy;
    same-cell wiring is ignored. This is an exploratory approximation because
    the graph does not currently retain physical-corridor provenance.
    """
    for node, data in graph.nodes(data=True):
        data["dg_size"] = 0.0
        data["dg_remote_endpoint_max"] = 0.0
        data.pop("dg_action_sizes", None)
        try:
            if int(node[2]) < 0:
                data["dg_action_sizes"] = {}
        except Exception:
            pass
    for _u, _v, data in graph.edges(data=True):
        data["dg_size"] = 0.0

    corridor = nx.Graph()
    shortcuts = []  # (node_u, node_v, cell_u, cell_v)
    for u, v in graph.edges():
        try:
            cu = (int(u[0]), int(u[1]))
            cv = (int(v[0]), int(v[1]))
        except Exception:
            continue
        if cu == cv:
            continue
        if abs(cu[0] - cv[0]) + abs(cu[1] - cv[1]) == 1:
            corridor.add_edge(cu, cv)
        else:
            shortcuts.append((u, v, cu, cv))
    for _u, _v, cu, cv in shortcuts:
        corridor.add_node(cu)
        corridor.add_node(cv)

    size_by_cells = {}
    for u, v, cu, cv in shortcuts:
        cell_key = tuple(sorted((cu, cv)))
        if cell_key not in size_by_cells:
            try:
                size_by_cells[cell_key] = float(
                    nx.shortest_path_length(corridor, cu, cv) + 1
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                size_by_cells[cell_key] = 0.0
        size = float(size_by_cells[cell_key])
        graph.edges[u, v]["dg_size"] = size

        for node in (u, v):
            try:
                if int(node[2]) >= 0:
                    prev = float(graph.nodes[node].get("dg_size", 0.0))
                    endpoint_max = max(prev, size)
                    # ``dg_size`` is kept for compatibility with the first v7
                    # graph dumps; policy readout never consumes this value.
                    graph.nodes[node]["dg_size"] = endpoint_max
                    graph.nodes[node]["dg_remote_endpoint_max"] = endpoint_max
            except Exception:
                pass

        # Project Q(current)--D(memory, action) onto Q(current)'s action gate.
        for query_node, direction_node in ((u, v), (v, u)):
            try:
                if int(query_node[2]) >= 0 or int(direction_node[2]) < 0:
                    continue
                action = int(direction_node[2])
                action_sizes = graph.nodes[query_node].setdefault(
                    "dg_action_sizes", {}
                )
                previous = float(action_sizes.get(action, action_sizes.get(str(action), 0.0)))
                action_sizes[action] = max(previous, size)
            except Exception:
                continue


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
