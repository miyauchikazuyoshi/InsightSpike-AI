"""Golden trace for the active, not-yet-migrated maze evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import networkx as nx
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_MAZE_ROOT = _REPO_ROOT / "experiments" / "maze"
if str(_MAZE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAZE_ROOT))


def _as_node(raw):
    return tuple(int(value) for value in raw)


def test_active_sp_evaluator_matches_frozen_golden_trace() -> None:
    """Freeze active behavior without claiming MazeFEval equivalence."""

    from insightspike.algorithms.gedig.config import GeDIGConfig
    from insightspike.algorithms.gedig_core import GeDIGCore
    from qhlib.evaluator import evaluate_multihop

    fixture_path = (
        Path(__file__).with_name("fixtures")
        / "maze_active_sp_trace.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    inputs = fixture["input"]
    expected = fixture["expected"]

    nodes = [_as_node(raw) for raw in inputs["path_nodes"]]
    before = nx.Graph()
    before.add_edges_from(zip(nodes, nodes[1:]))
    candidate = tuple(
        _as_node(raw)
        for raw in inputs["candidate_edge"]
    )
    config = GeDIGConfig(
        lambda_weight=1.0,
        sp_beta=0.5,
        sp_scope_mode="union",
        sp_hop_expand=0,
        sp_boundary_mode="induced",
        enable_spectral=False,
    )

    result = evaluate_multihop(
        core=GeDIGCore(config=config),
        prev_graph=before,
        stage_graph=before.copy(),
        g_before_for_expansion=before,
        anchors_core={nodes[0]},
        anchors_top_before={nodes[-1]},
        anchors_top_after={nodes[-1]},
        ecand=[(candidate[0], candidate[1], {})],
        base_ig=inputs["base_ig"],
        denom_cmax_base=inputs["denom_cmax_base"],
        max_hops=inputs["max_hops"],
        eval_all_hops=True,
        sp_signed=True,
        sp_pair_samples=0,
    )

    assert result.g0 == pytest.approx(expected["g0"])
    assert result.gmin == pytest.approx(expected["gmin"])
    assert result.best_hop == expected["best_hop"]
    assert result.delta_ged == pytest.approx(
        expected["delta_ged"]
    )
    assert result.delta_ig == pytest.approx(expected["delta_ig"])
    assert result.delta_sp == pytest.approx(expected["delta_sp"])
    assert [
        [list(start), list(end)]
        for start, end in result.chosen_edges_by_hop
    ] == expected["chosen_edges_by_hop"]

    assert len(result.hop_series) == len(expected["hop_series"])
    for actual, golden in zip(
        result.hop_series,
        expected["hop_series"],
    ):
        assert actual["hop"] == golden["hop"]
        assert actual["g"] == pytest.approx(golden["g"])
        assert actual["ged"] == pytest.approx(golden["ged"])
        assert actual["ig"] == pytest.approx(golden["ig"])
        assert actual["sp"] == pytest.approx(golden["sp"])
        assert actual["sp_pairs"] == golden["sp_pairs"]
