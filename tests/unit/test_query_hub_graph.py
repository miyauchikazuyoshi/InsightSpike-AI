from __future__ import annotations

from typing import List, Tuple

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "experiments" / "maze-query-hub-prototype" / "run_experiment_query.py"

if not MODULE_PATH.exists():
    raise RuntimeError(f"run_experiment_query.py not found at {MODULE_PATH}")

spec = importlib.util.spec_from_file_location("query_hub_runner", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("failed to build module spec for query hub runner")
query_hub_runner = importlib.util.module_from_spec(spec)
sys.modules.setdefault("query_hub_runner", query_hub_runner)
spec.loader.exec_module(query_hub_runner)

EpisodeArtifacts = query_hub_runner.EpisodeArtifacts
QueryHubConfig = query_hub_runner.QueryHubConfig
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
