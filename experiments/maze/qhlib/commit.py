from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

Node = Tuple[int, int, int]
Edge = Tuple[Node, Node]


def _direction_from_delta(delta: Tuple[int, int]) -> Optional[int]:
    # Keep in sync with run_experiment_query.DIR_TO_DELTA
    mapping = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    rev = {v: k for k, v in mapping.items()}
    try:
        dr, dc = int(delta[0]), int(delta[1])
        return rev.get((dr, dc))
    except Exception:
        return None


def _canonical_node_id(node: Sequence[int]) -> Node:
    if node is None:
        return (0, 0, -1)
    if len(node) == 3:
        return (int(node[0]), int(node[1]), int(node[2]))
    if len(node) == 2:
        return (int(node[0]), int(node[1]), -1)
    return (int(node[0]), 0, -1)


def _canonical_edge(a: Sequence[int], b: Sequence[int]) -> Edge:
    na = _canonical_node_id(a)
    nb = _canonical_node_id(b)
    return tuple(sorted([na, nb]))  # type: ignore[return-value]


def _ensure_dir_node_on(
    g: nx.Graph,
    anchor: Tuple[int, int],
    dir_idx: int,
    meta: Optional[Dict[str, Any]] = None,
) -> Node:
    node_id: Node = (int(anchor[0]), int(anchor[1]), int(dir_idx))
    if not g.has_node(node_id):
        attrs: Dict[str, Any] = {"node_type": "direction", "direction": int(dir_idx)}
        if meta:
            # pass-through a subset of useful attributes
            for k in ("vector", "abs_vector", "visit_count", "anchor_positions", "target_position", "birth_step", "source"):
                if k in meta:
                    attrs[k] = meta[k]
        g.add_node(node_id, **attrs)
    return node_id


def apply_commit_policy(
    *,
    base_graph: nx.Graph,
    current_query_node: Node,
    hop0_items: List[Dict[str, Any]],
    chosen_edges_by_hop: List[Tuple[Node, Node]],
    best_hop: int,
    policy: str = "threshold",
    fire_dg: bool = False,
    commit_budget: int = 0,
    bfs_shortcut: bool = False,
    # diagnostics/visualisation plumbing
    cand_edge_store: Optional[List[Tuple[Node, Node, bool, int]]] = None,
) -> Tuple[nx.Graph, List[List[List[int]]]]:
    """
    Apply two-phase commit policy on a copy of base_graph and return the committed graph
    and a list of committed edge snapshots for the step log.

    - Always commits hop0 (Top-L) Q↔dir edges.
    - For hop>0, commits up to best_hop edges only if policy allows (threshold/always) and
      only edges between the current Q and a direction node are applied.
    - commit_budget caps the number of additional edges committed (0 for no extra cap).
    """

    graph_commit = base_graph.copy()

    def _append_cand_edge(u: Node, v: Node, forced: bool, bridge: int) -> None:
        if cand_edge_store is not None:
            cand_edge_store.append((u, v, forced, bridge))

    # Commit hop0: Q↔dir for all items
    for item in hop0_items:
        anchor_src = item.get("anchor_position") or item.get("position") or []
        try:
            at = (int(anchor_src[0]), int(anchor_src[1]))
        except Exception:
            # If anchor missing, skip
            continue
        d = item.get("direction")
        if d is None:
            rd = tuple(item.get("relative_delta") or item.get("meta_delta") or (0, 0))
            d = _direction_from_delta(rd)
        if d is None:
            continue
        dir_node = _ensure_dir_node_on(graph_commit, at, int(d), item)
        if not graph_commit.has_edge(current_query_node, dir_node):
            graph_commit.add_edge(current_query_node, dir_node)
        # visualisation: stage=0 (base hop0 commit)
        _append_cand_edge(current_query_node, dir_node, bool(item.get("forced", False)), 0)

    # Policy gate for hop>0 commits
    pol = str(policy).lower()
    do_commit = (pol == "always") or (pol == "threshold" and bool(fire_dg))
    committed_snap: List[List[List[int]]] = []

    if do_commit and chosen_edges_by_hop:
        limit = best_hop if best_hop > 0 else 0
        to_commit = chosen_edges_by_hop[:limit] if limit > 0 else []
        cap = int(max(0, int(commit_budget)))
        if cap > 0:
            to_commit = to_commit[:cap]
        for eu, ev in to_commit:
            # By default: only allow (current Q) ↔ (direction). When bfs_shortcut=True,
            # commit any chosen edge (eu, ev) to realize multi-hop shortcut in one step.
            if not bfs_shortcut:
                try:
                    is_eu_q = int(eu[2]) == -1
                    is_ev_q = int(ev[2]) == -1
                except Exception:
                    is_eu_q = False; is_ev_q = False
                allow = (eu == current_query_node and not is_ev_q) or (ev == current_query_node and not is_eu_q)
                if not allow:
                    continue
            if not graph_commit.has_node(eu):
                graph_commit.add_node(eu)
            if not graph_commit.has_node(ev):
                graph_commit.add_node(ev)
            if not graph_commit.has_edge(eu, ev):
                graph_commit.add_edge(eu, ev)
            _append_cand_edge(eu, ev, False, 1)
            committed_snap.append([[int(eu[0]), int(eu[1]), int(eu[2])], [int(ev[0]), int(ev[1]), int(ev[2])]])

    return graph_commit, committed_snap
