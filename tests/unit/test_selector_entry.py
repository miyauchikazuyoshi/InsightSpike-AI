import networkx as nx


def test_selector_pure_full_ab_smoke():
    from insightspike.algorithms.gedig.selector import compute_gedig

    g_prev = nx.Graph(); g_prev.add_edge(1, 2)
    g_curr = nx.Graph(); g_curr.add_edges_from([(1, 2), (2, 3)])

    pure = compute_gedig(g_prev, g_curr, mode="pure")
    assert isinstance(pure, dict) and pure.get("mode") == "pure"
    assert set(pure.keys()) >= {"gedig", "ged", "ig"}

    full = compute_gedig(g_prev, g_curr, mode="full")
    assert isinstance(full, dict) and full.get("mode") == "full"
    assert set(full.keys()) >= {"gedig", "ged", "ig"}

    ab = compute_gedig(g_prev, g_curr, mode="ab", variant="B")
    assert isinstance(ab, dict) and ab.get("mode") == "ab"
    assert "pure" in ab and "full" in ab and ab.get("ab", {}).get("variant") == "B"

