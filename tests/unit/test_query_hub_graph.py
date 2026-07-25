from __future__ import annotations

from typing import List, Tuple

import importlib.util
import json
import math
import sys
from pathlib import Path

import networkx as nx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Path was renamed in the 2026 maze refactor (maze-query-hub-prototype → maze).
# Try both for backward compatibility.
_CANDIDATE_PATHS = [
    PROJECT_ROOT / "experiments" / "maze" / "run_experiment_query.py",
    PROJECT_ROOT / "experiments" / "maze-query-hub-prototype" / "run_experiment_query.py",
]
MODULE_PATH = next((p for p in _CANDIDATE_PATHS if p.exists()), None)

if MODULE_PATH is None:
    pytest.skip(
        f"run_experiment_query.py not found in any of: {[str(p) for p in _CANDIDATE_PATHS]}",
        allow_module_level=True,
    )

# Add experiments/maze to sys.path so internal `from qhlib...` imports resolve.
_MAZE_DIR = str(MODULE_PATH.parent)
if _MAZE_DIR not in sys.path:
    sys.path.insert(0, _MAZE_DIR)

spec = importlib.util.spec_from_file_location("query_hub_runner", MODULE_PATH)
if spec is None or spec.loader is None:
    pytest.skip("failed to build module spec for query hub runner", allow_module_level=True)
query_hub_runner = importlib.util.module_from_spec(spec)
sys.modules.setdefault("query_hub_runner", query_hub_runner)
try:
    spec.loader.exec_module(query_hub_runner)
except Exception as e:  # pragma: no cover — module-load issues skip rather than fail collection
    pytest.skip(f"could not import query_hub_runner: {e}", allow_module_level=True)

EpisodeArtifacts = query_hub_runner.EpisodeArtifacts
QueryHubConfig = query_hub_runner.QueryHubConfig
annotate_dg_size = query_hub_runner._annotate_dg_size
dg_action_log_bias = query_hub_runner.dg_action_log_bias
dg_action_multiplier = query_hub_runner.dg_action_multiplier
dg_action_signal = query_hub_runner.dg_action_signal
make_query_node = query_hub_runner.make_query_node
run_episode_query = query_hub_runner.run_episode_query


def _make_config(max_steps: int = 10) -> QueryHubConfig:
    return QueryHubConfig(
        maze_size=5,
        maze_type="dfs",
        max_steps=max_steps,
        selector={
            "theta_cand": 1.0,
            "theta_link": 0.1,
            "candidate_cap": 32,
            "top_m": 32,
            "cand_radius": 1.0,
            "link_radius": 0.1,
        },
        gedig={
            "lambda_weight": 0.5,
            "max_hops": 3,
            "decay_factor": 0.7,
            "adaptive_hops": False,
        },
    )


def _run_episode(seed: int = 0, max_steps: int = 10) -> EpisodeArtifacts:
    return run_episode_query(seed=seed, config=_make_config(max_steps=max_steps))


def test_query_nodes_are_recorded_with_marker() -> None:
    artifacts = _run_episode()
    assert artifacts.steps, "expected at least one step record"

    for step in artifacts.steps:
        assert len(step.query_node) == 3
        assert step.query_node[2] == -1, "query node must use marker -1"

        query_nodes = [
            tuple(node)
            for node in step.graph_nodes
            if isinstance(node, list) and len(node) == 3 and node[2] == -1
        ]
        assert (
            query_nodes
        ), f"expected query node snapshot in graph_nodes at step {step.step}"
        assert len(step.query_vector) == 8, "query vector should be 8D"


def test_candidate_targets_are_present_and_cand_pool_matches_target_positions() -> None:
    artifacts = _run_episode()
    for step in artifacts.steps:
        for item in step.candidate_pool:
            target = item.get("target_position") or item.get("targetPosition")
            assert target is not None, f"candidate missing target_position at step {step.step}"
            assert len(target) == 2
            if item.get("origin") == "obs":
                anchor = item.get("position") or item.get("pos")
                assert anchor is not None
                assert list(anchor) != list(
                    target
                ), "observation candidate should point to next cell, not anchor"

        for item in step.selected_links:
            target = item.get("target_position") or item.get("targetPosition")
            assert target is not None
            assert len(target) == 2


def test_query_node_progresses_with_agent_motion() -> None:
    artifacts = _run_episode(max_steps=20)
    positions: List[Tuple[int, int]] = [
        (int(node.query_node[0]), int(node.query_node[1])) for node in artifacts.steps
    ]
    unique_positions = set(positions)
    assert (
        len(unique_positions) >= 2
    ), "expected query node to move across at least two distinct cells"


def _make_dg_config(policy: str, alpha: float) -> QueryHubConfig:
    config = QueryHubConfig(
        maze_size=20,
        maze_type="simple",
        max_steps=1,
        selector={
            "theta_cand": 1.0,
            "theta_link": 0.1,
            "candidate_cap": 32,
            "top_m": 32,
            "cand_radius": 1.0,
            "link_radius": 0.1,
        },
        gedig={
            "lambda_weight": 0.5,
            "max_hops": 0,
            "decay_factor": 0.7,
            "adaptive_hops": False,
        },
    )
    config.action_policy = policy
    config.action_temp = 1.0
    config.advantage_commit = 1.01
    config.propagated_alpha = 0.0
    config.dg_action_alpha = alpha
    config.dg_action_scale = 10.0
    config.snapshot_level = "minimal"
    config.post_sp_diagnostics = False
    return config


def test_dg_action_signal_uses_source_query_projection_and_safe_parameters() -> None:
    graph = nx.Graph()
    query = make_query_node((1, 1))
    graph.add_node(query, dg_action_sizes={"2": 10.0})

    assert dg_action_signal(graph, (1, 1), 2, 10.0) == pytest.approx(math.tanh(1.0))
    assert dg_action_signal(graph, (1, 1), 1, 10.0) == 0.0
    for invalid_scale in (0.0, -1.0, math.inf, math.nan):
        assert dg_action_signal(graph, (1, 1), 2, invalid_scale) == 0.0

    graph.nodes[query]["dg_action_sizes"] = {2: math.nan}
    assert dg_action_signal(graph, (1, 1), 2, 10.0) == 0.0


def test_dg_annotation_to_action_readout_contract() -> None:
    graph = nx.Graph()
    query = (0, 0, -1)
    midpoint = (0, 1, -1)
    recalled_direction = (0, 2, 2)
    graph.add_edges_from(
        [
            (query, midpoint),
            (midpoint, recalled_direction),
            (query, recalled_direction),
        ]
    )

    annotate_dg_size(graph)

    assert graph.edges[query, recalled_direction]["dg_size"] == pytest.approx(3.0)
    assert graph.nodes[query]["dg_action_sizes"][2] == pytest.approx(3.0)
    assert dg_action_signal(graph, (0, 0), 2, 10.0) == pytest.approx(
        math.tanh(0.3)
    )


def test_dg_action_alpha_zero_is_exact_noop_and_log_bias_is_bounded() -> None:
    graph = nx.Graph()
    graph.add_node(make_query_node((1, 1)), dg_action_sizes={2: 100.0})

    assert dg_action_log_bias(graph, (1, 1), 2, 0.0, 10.0) == 0.0
    assert dg_action_multiplier(graph, (1, 1), 2, 0.0, 10.0) == 1.0
    assert dg_action_log_bias(graph, (1, 1), 2, math.inf, 10.0) == 0.0
    assert dg_action_log_bias(graph, (1, 1), 2, 1e9, 10.0) == 60.0


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--dg-action-alpha", "nan"),
        ("--dg-action-alpha", "inf"),
        ("--dg-action-scale", "0"),
        ("--dg-action-scale", "-1"),
    ],
)
def test_cli_rejects_invalid_dg_parameters(
    flag: str,
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_experiment_query.py", flag, value])
    with pytest.raises(SystemExit):
        query_hub_runner.parse_args()


@pytest.mark.parametrize("policy", ["argmax", "softmax"])
def test_dg_action_gate_changes_controlled_one_step_choice(policy: str) -> None:
    baseline = run_episode_query(seed=0, config=_make_dg_config(policy, alpha=0.0))
    baseline_step = baseline.steps[0]
    action_by_name = {
        name: action
        for action, name in query_hub_runner.SimpleMaze.ACTION_NAMES.items()
    }
    baseline_action = action_by_name[baseline_step.action]
    target_action = next(
        action for action in baseline_step.possible_moves if action != baseline_action
    )

    graph = nx.Graph()
    graph.add_node(
        make_query_node((1, 1)),
        dg_action_sizes={target_action: 100.0},
    )

    control = run_episode_query(
        seed=0,
        config=_make_dg_config(policy, alpha=0.0),
        inherited_graph=graph,
    )
    active = run_episode_query(
        seed=0,
        config=_make_dg_config(policy, alpha=20.0),
        inherited_graph=graph,
    )

    assert control.steps[0].action == baseline_step.action
    assert control.steps[0].dg_action_feasible_candidate_count == 1
    assert control.steps[0].dg_action_eligible_action_count == 2
    assert control.steps[0].dg_action_candidate_count == 1
    assert not control.steps[0].dg_action_enabled
    assert not control.steps[0].dg_action_applied
    assert active.steps[0].action == query_hub_runner.SimpleMaze.ACTION_NAMES[target_action]
    assert active.steps[0].dg_action_exposed
    assert active.steps[0].dg_action_competitive
    assert active.steps[0].dg_action_applied
    assert active.steps[0].dg_action_signal_spread > 0.0
    assert active.steps[0].dg_action_log_bias_spread > 0.0
    assert active.steps[0].dg_action_log_bias > 0.0
    assert active.summary["dg_action_competitive_steps"] == 1


def test_dg_telemetry_excludes_anti_backtrack_masked_signal() -> None:
    config = _make_dg_config("argmax", alpha=0.0)
    config.max_steps = 2
    baseline = run_episode_query(seed=0, config=config)
    first_action_by_name = {
        name: action
        for action, name in query_hub_runner.SimpleMaze.ACTION_NAMES.items()
    }
    first_action = first_action_by_name[baseline.steps[0].action]
    backtrack_action = {0: 2, 1: 3, 2: 0, 3: 1}[first_action]
    second_position = tuple(baseline.steps[1].query_node_pre[:2])

    graph = nx.Graph()
    graph.add_node(
        make_query_node(second_position),
        dg_action_sizes={backtrack_action: 100.0},
    )
    active_config = _make_dg_config("argmax", alpha=20.0)
    active_config.max_steps = 2
    active = run_episode_query(
        seed=0,
        config=active_config,
        inherited_graph=graph,
    )
    second = active.steps[1]

    assert second.dg_action_feasible_candidate_count == 1
    assert second.dg_action_candidate_count == 0
    assert not second.dg_action_exposed
    assert not second.dg_action_competitive
    assert not second.dg_action_applied


def test_cli_output_records_dg_provenance_and_minimal_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result.json"
    steps = tmp_path / "steps.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment_query.py",
            "--maze-size",
            "20",
            "--maze-type",
            "simple",
            "--max-steps",
            "1",
            "--max-hops",
            "0",
            "--sp-ds-sqlite",
            "",
            "--steps-ultra-light",
            "--dg-action-alpha",
            "2.5",
            "--dg-action-scale",
            "7",
            "--step-log",
            str(steps),
            "--output",
            str(output),
        ],
    )

    query_hub_runner.main()

    result = json.loads(output.read_text(encoding="utf-8"))
    dg_config = result["config"]["graph_persistent_dg"]
    assert dg_config["dg_action_alpha"] == pytest.approx(2.5)
    assert dg_config["dg_action_scale"] == pytest.approx(7.0)
    assert dg_config["dg_projection"] == "source-query-action"
    assert dg_config["dg_telemetry_version"] == 2

    incremental = steps.with_suffix(".incremental.jsonl")
    row = json.loads(incremental.read_text(encoding="utf-8").splitlines()[0])
    assert not row["dg_action_enabled"]
    assert not row["dg_action_exposed"]
    assert not row["dg_action_competitive"]
    assert not row["dg_action_applied"]
    assert row["dg_action_feasible_candidate_count"] == 0
    assert row["dg_action_candidate_count"] == 0
    assert row["dg_action_value"] == 0.0
    assert row["dg_action_log_bias"] == 0.0
