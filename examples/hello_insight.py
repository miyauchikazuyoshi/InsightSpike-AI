"""Minimal geDIG gauge demo.

Builds two tiny graphs and computes F = ΔEPC_norm − λ·ΔIG.
Prints F, ΔEPC_norm, ΔIG, and spike flag.
"""
from __future__ import annotations

import networkx as nx
from insightspike.algorithms.gedig_core import GeDIGCore
from insightspike.algorithms.linkset_adapter import build_linkset_info


def tiny_graphs():
    # Before: chain 3 nodes (A-B-C)
    g_before = nx.Graph()
    g_before.add_edges_from([("A", "B"), ("B", "C")])

    # After: add a shortcut (A-C)
    g_after = nx.Graph()
    g_after.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
    return g_before, g_after


def main() -> None:
    g1, g2 = tiny_graphs()
    core = GeDIGCore(
        enable_multihop=False,  # keep it simple
        lambda_weight=1.0,
        spike_threshold=-0.2,   # detect relatively easy
        ig_mode="norm",         # normalized IG for readability
    )
    # Minimal linkset_info for paper-aligned IG
    ls = build_linkset_info(
        s_link=[{"index": 1, "similarity": 1.0}],
        candidate_pool=[],
        decision={"index": 1, "similarity": 1.0},
        query_vector=[1.0],
        base_mode="link",
    )
    res = core.calculate(g_prev=g1, g_now=g2, linkset_info=ls)
    print(
        f"F = {res.gedig_value:.3f}  "
        f"(ΔEPC_norm={res.delta_ged_norm:.3f},  ΔIG={res.ig_value:.3f},  spike={res.spike})"
    )


if __name__ == "__main__":
    main()
