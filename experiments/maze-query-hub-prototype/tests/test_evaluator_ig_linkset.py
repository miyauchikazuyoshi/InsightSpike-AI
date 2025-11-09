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


def test_ig_recompute_with_linkset_constant_across_hops() -> None:
    # Build trivial graphs; linkset metrics do not depend on the graphs in current core
    prev_graph = nx.Graph()
    stage_graph = nx.Graph()

    a = _n(0, 0, 0)
    b = _n(0, 1, 0)
    prev_graph.add_node(a); prev_graph.add_node(b)
    stage_graph.add_node(a); stage_graph.add_node(b)

    g_before_for_expansion = prev_graph

    anchors_core: Set[Node] = {a}
    anchors_top_before: Set[Node] = {a, b}
    anchors_top_after: Set[Node] = {a, b}

    # Linkset payload with two base items and a query; similarities all positive
    pre_linkset_info = {
        "s_link": [
            {"index": 1, "similarity": 1.0},
            {"index": 2, "similarity": 1.0},
        ],
        "candidate_pool": [],
        "decision": {"index": 1, "similarity": 1.0},
        "query_entry": {"index": "query", "similarity": 1.0},
        "base_mode": "link",
    }

    # Strict/paper linksetを有効化し、ホップ毎にlinkset IGを採用
    core = GeDIGCore(enable_multihop=True, max_hops=1, lambda_weight=1.0, sp_beta=0.0, linkset_mode=True, ig_source_mode="linkset", ig_hop_apply="all")

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
        pre_linkset_info=pre_linkset_info,
        query_vec=None,
        ig_fixed_den=None,
        eval_all_hops=True,
        sp_early_stop=False,
        sp_cache=False,
    )

    assert len(res.hop_series) >= 2
    hop0 = next(h for h in res.hop_series if h.get("hop") == 0)
    hop1 = next(h for h in res.hop_series if h.get("hop") == 1)
    # hop0はbase_ig適用（0.0）でIG=0、hop1はlinkset IGが適用され非ゼロになる
    assert abs(float(hop0["ig"])) < 1e-9
    assert isinstance(hop1["ig"], float) and abs(hop1["ig"]) > 0.0

