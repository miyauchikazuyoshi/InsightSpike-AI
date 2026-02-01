from __future__ import annotations

"""
EdgeBuilder (stub for Phase 2)

Responsible for assembling candidate sets:
  - S_link (Top‑L)
  - forced fallback (unvisited→nearest dir)
  - Ecand for greedy multi-hop evaluation (mem-only, prev_graph-existing)

The legacy runner will be incrementally migrated to call these builders.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx

Node = Tuple[int, int, int]
QUERY_MARKER = -1


DIR_TO_DELTA = {
    0: (-1, 0),  # north
    1: (0, 1),   # east
    2: (1, 0),   # south
    3: (0, -1),  # west
}
DELTA_TO_DIR = {delta: direction for direction, delta in DIR_TO_DELTA.items()}


def direction_from_delta(delta: Tuple[int, int]) -> Optional[int]:
    try:
        return DELTA_TO_DIR.get((int(delta[0]), int(delta[1])))
    except Exception:
        return None


def make_direction_node(anchor: Tuple[int, int], direction: int) -> Node:
    return (int(anchor[0]), int(anchor[1]), int(direction))


def canonical_node_id(node: Sequence[int]) -> Node:
    if len(node) == 3:
        return (int(node[0]), int(node[1]), int(node[2]))
    if len(node) == 2:
        return (int(node[0]), int(node[1]), QUERY_MARKER)
    return (int(node[0]), 0, QUERY_MARKER)


def canonical_edge_id(a: Sequence[int], b: Sequence[int]) -> Tuple[Node, Node]:
    na = canonical_node_id(a)
    nb = canonical_node_id(b)
    return tuple(sorted([na, nb]))  # type: ignore[return-value]


def build_ecand(
    *,
    prev_graph: nx.Graph,
    selection_candidates: Iterable[Dict[str, Any]],
    current_query_node: Node,
    anchor_position: Tuple[int, int],
    cap_topk: int = 0,
    include_qpast: bool = True,
    ring_center: Optional[Tuple[int, int]] = None,
    ring_size: Optional[Tuple[int, int]] = None,
    ellipse: bool = False,
    # Optional: prefilter by explicit node-id set (e.g., weighted-L2 index results)
    prefilter_nodes: Optional[Set[Node]] = None,
) -> Tuple[List[Tuple[Node, Node, Dict[str, Any]]], int, int]:
    """Build Ecand for greedy SP evaluation.

    - From memory-origin direction candidates whose direction node exists in prev_graph
    - Optionally include Q→pastQ edges from prev_graph
    - Cap by cap_topk for speed control

    Returns: (ecand_edges, mem_count, qpast_count)
    """
    ecand: List[Tuple[Node, Node, Dict[str, Any]]] = []
    seen: Set[Tuple[Node, Node]] = set()
    mem_count = 0
    qpast_count = 0

    # Optional ring prefilter for mem-origin items (skipped when prefilter_nodes is provided)
    def _in_ring(item_anchor: Tuple[int, int]) -> bool:
        if prefilter_nodes is not None:
            return True
        if ring_center is None or ring_size is None:
            return True
        ar, ac = int(item_anchor[0]), int(item_anchor[1])
        cr, cc = int(ring_center[0]), int(ring_center[1])
        Rr, Rc = max(0, int(ring_size[0])), max(0, int(ring_size[1]))
        dr = abs(ar - cr); dc = abs(ac - cc)
        if not ellipse:
            return (dr <= Rr) and (dc <= Rc)
        # ellipse: (dr/Rr)^2 + (dc/Rc)^2 <= 1 (guard div-by-zero)
        Rr_eff = max(1, Rr); Rc_eff = max(1, Rc)
        return (dr * dr) * (Rc_eff * Rc_eff) + (dc * dc) * (Rr_eff * Rr_eff) <= (Rr_eff * Rr_eff) * (Rc_eff * Rc_eff)

    for it in selection_candidates:
        origin = str(it.get("origin", "")).lower()
        if origin != "mem":
            continue
        anchor_src = it.get("anchor_position") or it.get("position") or [anchor_position[0], anchor_position[1]]
        at = (int(anchor_src[0]), int(anchor_src[1]))
        if not _in_ring(at):
            continue
        d = it.get("direction")
        if d is None:
            rd = tuple(it.get("relative_delta") or it.get("meta_delta") or (0, 0))
            d = direction_from_delta(rd)
        if d is None:
            continue
        nid_prev = make_direction_node(at, int(d))
        # Optional node-id prefilter (e.g., weighted-L2 radius search results)
        if prefilter_nodes is not None and nid_prev not in prefilter_nodes:
            continue
        if nid_prev not in prev_graph:
            # require existence on prev_graph so Lb>0 is meaningful
            continue
        e = canonical_edge_id(current_query_node, nid_prev)
        if e in seen:
            continue
        seen.add(e)
        ecand.append((e[0], e[1], dict(it)))
        mem_count += 1

    if include_qpast:
        try:
            for node_id, data in prev_graph.nodes(data=True):
                if data.get("node_type") == "query" and node_id != current_query_node:
                    e2 = canonical_edge_id(current_query_node, node_id)
                    if e2 not in seen:
                        seen.add(e2)
                        meta = {
                            "origin": "qpast",
                            "index": f"qpast:{node_id[0]},{node_id[1]}",
                            "anchor_position": [int(node_id[0]), int(node_id[1])],
                        }
                        ecand.append((e2[0], e2[1], meta))
                        qpast_count += 1
        except Exception:
            pass

    if cap_topk and cap_topk > 0 and len(ecand) > cap_topk:
        ecand = ecand[:cap_topk]

    return ecand, mem_count, qpast_count
