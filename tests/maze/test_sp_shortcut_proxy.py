import networkx as nx

from insightspike.algorithms.gedig_core import GeDIGCore


def test_sp_and_ged_min_proxy_shortcut():
    # Small contrived graph: 4-node path vs path+shortcut
    g_before = nx.Graph()
    g_before.add_edges_from([(0, 1), (1, 2), (2, 3)])
    g_after = g_before.copy()
    g_after.add_edge(0, 3)  # shortcut

    core = GeDIGCore(
        enable_ged_min_diag=True,
        sp_beta=1.0,
        use_multihop_sp_gain=True,
        sp_eval_mode="fixed_before_pairs",
        sp_scope_mode="union",
    )
    anchors = {0, 1, 2, 3}
    res = core.calculate(
        g_prev=g_before,
        g_now=g_after,
        features_prev=None,
        features_now=None,
        focal_nodes=anchors,
        k_star=4,  # ensures ig_fixed_den>0
        force_sp_gain_eval=True,
    )

    # Average shortest path drops from ~1.66 (path) to ~1.33 (cycle), gain ≈ 0.2
    assert res.delta_sp_rel > 0.15, f"delta_sp_rel was {res.delta_sp_rel}"
    # GED_min proxy should reflect positive compression (ASP gain > 0)
    assert res.ged_min_proxy > 0.0
