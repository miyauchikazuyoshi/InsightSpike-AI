"""Minimal example: compute g0/gmin and AG/DG."""
from __future__ import annotations

import networkx as nx

from insightspike.algorithms.gedig_core import GeDIGCore
from insightspike.algorithms.linkset_adapter import build_linkset_info
from insightspike.algorithms.gating import decide_gates


def tiny_graphs():
    g_before = nx.Graph()
    g_before.add_edges_from([("A", "B"), ("B", "C")])
    g_after = nx.Graph()
    g_after.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
    return g_before, g_after


def main() -> None:
    g1, g2 = tiny_graphs()
    core = GeDIGCore(enable_multihop=True, max_hops=2, lambda_weight=1.0, ig_mode="norm")
    ls = build_linkset_info(
        s_link=[{"index": 1, "similarity": 1.0}],
        candidate_pool=[],
        decision={"index": 1, "similarity": 1.0},
        query_vector=[1.0],
        base_mode="link",
    )
    res = core.calculate(g_prev=g1, g_now=g2, linkset_info=ls)

    if res.hop_results and 0 in res.hop_results:
        g0 = float(res.hop_results[0].gedig)
    else:
        g0 = float(res.gedig_value)
    gmin = float(res.gedig_value)

    gate = decide_gates(g0=g0, gmin=gmin, theta_ag=0.5, theta_dg=0.0)
    print(f"g0={gate.g0:.3f}, gmin={gate.gmin:.3f}, AG={gate.ag}, DG={gate.dg}, b(t)={gate.b_value:.3f}")


if __name__ == "__main__":
    main()
