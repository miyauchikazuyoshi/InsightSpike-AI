from typing import Dict, Any, List, Set, Tuple

import networkx as nx

from insightspike.algorithms.gedig_core import GeDIGCore
import importlib.util as _imp_util
import sys as _sys
from pathlib import Path as _Path

_evaluator_path = _Path(__file__).resolve().parents[1] / "qhlib" / "evaluator.py"
_spec = _imp_util.spec_from_file_location("qh_evaluator", str(_evaluator_path))
assert _spec and _spec.loader, "failed to load evaluator.py"
_mod = _imp_util.module_from_spec(_spec)
_sys.modules["qh_evaluator"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
evaluate_multihop = getattr(_mod, "evaluate_multihop")

Node = Tuple[int, int, int]


def _n(r: int, c: int, d: int = 0) -> Node:
    return (int(r), int(c), int(d))


def test_ig_recompute_fallback_uses_entropy_when_no_linkset() -> None:
    # Small before/after graphs with distinct node features
    prev_graph = nx.Graph()
    stage_graph = nx.Graph()

    a = _n(0, 0, 0)
    b = _n(0, 1, 0)

    prev_graph.add_node(a, vector=[0.0] * 8)
    prev_graph.add_node(b, vector=[0.0] * 8)
    # After graph has a different feature for b
    stage_graph.add_node(a, vector=[0.0] * 8)
    stage_graph.add_node(b, vector=[10.0] + [0.0] * 7)

    # Expansion graph equals prev for this test
    g_before_for_expansion = prev_graph

    anchors_core: Set[Node] = {a}
    # Ensure both nodes are in the k-hop neighborhoods for consistency
    anchors_top_before: Set[Node] = {a, b}
    anchors_top_after: Set[Node] = {a, b}

    core = GeDIGCore(enable_multihop=True, max_hops=1, lambda_weight=1.0, sp_beta=0.0)

    res = evaluate_multihop(
        core=core,
        prev_graph=prev_graph,
        stage_graph=stage_graph,
        g_before_for_expansion=g_before_for_expansion,
        anchors_core=anchors_core,
        anchors_top_before=anchors_top_before,
        anchors_top_after=anchors_top_after,
        ecand=[],
        base_ig=0.0,
        denom_cmax_base=1.0,
        max_hops=1,
        ig_recompute=True,
        pre_linkset_info=None,
        query_vec=[1.0] + [0.0] * 7,
        ig_fixed_den=None,
        eval_all_hops=True,
        sp_early_stop=False,
        sp_cache=False,
    )

    # hop_series contains hop0 and hop1; hop0 IG equals base_ig (0.0), hop1 should be recomputed (> 0 or < 0)
    assert len(res.hop_series) >= 2
    hop0 = next(h for h in res.hop_series if h.get("hop") == 0)
    hop1 = next(h for h in res.hop_series if h.get("hop") == 1)
    assert abs(float(hop0["ig"])) < 1e-9
    # Ensure fallback path produced a finite IG value for hop1
    assert isinstance(hop1["ig"], float)
    assert hop1["ig"] == hop1["h"]  # sp_beta=0.0 so ig == ΔH
