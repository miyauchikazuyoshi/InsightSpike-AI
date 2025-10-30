from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


class L1Conductor:
    """Layer1 conductor for centers/Ecand orchestration (bridge for experiments).

    This helper builds a lightweight graph from documents (using ScalableGraphBuilder)
    and proposes candidate edges from graph.x for given centers. It can also produce
    per-hop candidates by intersecting with union-of-k-hop node sets around centers.
    """

    def __init__(self, config: Any | None = None) -> None:
        self.config = config

    def propose_candidates(
        self,
        documents: List[Dict[str, Any]],
        centers: Sequence[int],
        *,
        top_k: int = 16,
        theta_link: float = 0.3,
        max_hops: int = 3,
    ) -> Dict[str, Any]:
        from .scalable_graph_builder import ScalableGraphBuilder
        from ...metrics.pyg_compatible_metrics import pyg_to_networkx

        gb = ScalableGraphBuilder(self.config)
        graph = gb.build_graph(documents)

        # candidate_edges from cosine similarity over x
        cand = gb.propose_candidate_edges_from_graph(
            graph=graph, centers=list(centers), top_k=top_k, theta_link=theta_link
        )

        # by-hop candidates using union-of-k-hop over nx conversion
        nxg = pyg_to_networkx(graph)
        cand_by_hop: Dict[int, List[Tuple[int, int, Dict[str, Any]]]] = {}
        try:
            from collections import deque

            def k_hop_nodes(G, srcs: Sequence[int], k: int) -> set[int]:
                seen: set[int] = set()
                dq = deque()
                for s in srcs:
                    if s in G:
                        seen.add(int(s)); dq.append((int(s), 0))
                while dq:
                    u, d = dq.popleft()
                    if d >= k:
                        continue
                    for v in G.neighbors(u):
                        iv = int(v)
                        if iv not in seen:
                            seen.add(iv); dq.append((iv, d+1))
                return seen

            for h in range(0, max(0, int(max_hops)) + 1):
                nodes_h = k_hop_nodes(nxg, list(centers), h)
                hop_list: List[Tuple[int, int, Dict[str, Any]]] = []
                for (u, v, meta) in cand:
                    if (u in nodes_h) and (v in nodes_h):
                        hop_list.append((u, v, meta))
                cand_by_hop[h] = hop_list
        except Exception:
            cand_by_hop = {}

        return {
            'graph': graph,
            'candidate_edges': cand,
            'candidate_edges_by_hop': cand_by_hop or None,
        }

