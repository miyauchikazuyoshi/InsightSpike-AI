from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, Set

import numpy as np

# Match A/B's feature weighting for maze 8D vectors
WEIGHT_VECTOR = np.array([1.0, 1.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0], dtype=np.float32)


class _LiteData:
    def __init__(self, x: np.ndarray, edge_index: np.ndarray):
        self.x = x
        self.edge_index = edge_index
        self.num_nodes = int(x.shape[0])


def _nx_to_litedata(g, node_order: Sequence[Any], default_dim: int = 8) -> _LiteData:
    # Build features from node attributes: prefer 'abs_vector' or 'vector'
    feats = []
    for n in node_order:
        data = g.nodes[n]
        vec = data.get("abs_vector") or data.get("vector")
        if vec is None:
            arr = np.zeros(default_dim, dtype=np.float32)
        else:
            arr = np.asarray(vec, dtype=np.float32).flatten()
            if arr.size < default_dim:
                arr = np.pad(arr, (0, default_dim - arr.size))
            elif arr.size > default_dim:
                arr = arr[:default_dim]
        feats.append(arr.astype(np.float32))
    x = np.vstack(feats) if feats else np.zeros((0, default_dim), dtype=np.float32)
    # Apply weighting to align with Core's entropy_ig path
    if x.size and x.shape[1] == WEIGHT_VECTOR.size:
        x = x * WEIGHT_VECTOR
    # Edge index from mapping
    idx = {n: i for i, n in enumerate(node_order)}
    edges = []
    for u, v in g.edges():
        iu = idx.get(u); iv = idx.get(v)
        if iu is None or iv is None:
            continue
        edges.append([iu, iv])
        edges.append([iv, iu])  # undirected
    ei = np.asarray(edges, dtype=np.int64).T if edges else np.empty((2, 0), dtype=np.int64)
    return _LiteData(x=x, edge_index=ei)


def eval_query_centric_via_l3(
    *,
    prev_graph,  # nx.Graph
    curr_graph,  # nx.Graph
    centers: Sequence[Any] | None,
    cand_edges: Sequence[Tuple[Any, Any, Dict[str, Any]]] | None,
    sp_engine: str = "cached_incr",
    pair_samples: int = 200,
    budget: int = 1,
    cand_topk: int = 0,
    default_dim: int = 8,
    max_hops: int = 3,
) -> Dict[str, Any]:
    """Use main L3GraphReasoner to compute query-centric metrics between two NX graphs.

    Returns an object with 'metrics' mirroring L3.analyze_documents output.
    """
    import os
    # Apply ENV knobs (allow caller to override per-call)
    os.environ['INSIGHTSPIKE_SP_ENGINE'] = str(sp_engine)
    os.environ['INSIGHTSPIKE_SP_PAIR_SAMPLES'] = str(int(max(1, pair_samples)))
    os.environ['INSIGHTSPIKE_SP_BUDGET'] = str(int(max(0, budget)))
    if cand_topk and int(cand_topk) > 0:
        os.environ['INSIGHTSPIKE_CAND_TOPK'] = str(int(cand_topk))

    # Build node order and features consistently for both graphs
    node_order = list(curr_graph.nodes())
    data_prev = _nx_to_litedata(prev_graph, node_order=node_order, default_dim=default_dim)
    data_curr = _nx_to_litedata(curr_graph, node_order=node_order, default_dim=default_dim)

    # Map centers and candidate edges to index space
    idx_map = {n: i for i, n in enumerate(node_order)}
    centers_idx: List[int] = []
    if centers:
        for c in centers:
            i = idx_map.get(c)
            if i is not None:
                centers_idx.append(int(i))
    # candidate edges as (u_idx, v_idx, meta)
    cand_idx = None
    if cand_edges:
        tmp = []
        for (u, v, meta) in cand_edges:
            iu = idx_map.get(u); iv = idx_map.get(v)
            if iu is None or iv is None:
                continue
            tmp.append((int(iu), int(iv), meta or {}))
        cand_idx = tmp

    # Optional: by-hop candidates (union-of-k-hop)
    cand_by_hop_idx: Dict[int, List[Tuple[int,int,Dict[str,Any]]]] | None = None
    try:
        import networkx as nx
        # expand from centers over curr_graph
        def k_hop_nodes(G: Any, srcs: Sequence[Any], k: int) -> set[int]:
            seen: set[int] = set()
            from collections import deque
            dq = deque()
            for c in srcs:
                if c in G:
                    seen.add(int(c)); dq.append((int(c), 0))
            while dq:
                u, d = dq.popleft()
                if d >= k:
                    continue
                for v in G.neighbors(u):
                    iv = int(v)
                    if iv not in seen:
                        seen.add(iv); dq.append((iv, d+1))
            return seen
        cand_by_hop_idx = {}
        centers_set = set(centers or [])
        for h in range(0, max(0, int(max_hops)) + 1):
            nodes_h = k_hop_nodes(curr_graph, list(centers_set), h)
            hop_list: List[Tuple[int,int,Dict[str,Any]]] = []
            if cand_idx:
                for (u,v,meta) in cand_idx:
                    if (u in nodes_h) and (v in nodes_h):
                        hop_list.append((u, v, meta))
            cand_by_hop_idx[h] = hop_list
    except Exception:
        cand_by_hop_idx = None

    # Call L3
    from insightspike.implementations.layers.layer3_graph_reasoner import L3GraphReasoner
    l3 = L3GraphReasoner()
    context = {
        'graph': data_curr,
        'previous_graph': data_prev,
        'centers': centers_idx,
        'candidate_edges': cand_idx,
        'candidate_edges_by_hop': cand_by_hop_idx,
    }
    res = l3.analyze_documents([], context)  # docs unused when graph provided
    return res
