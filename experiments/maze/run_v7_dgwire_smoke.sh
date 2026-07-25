#!/usr/bin/env bash
# v7 DG-action wiring smoke: alpha=0 control vs alpha=3 active.
# This is an exploratory, single-seed manipulation check, not a confirmatory
# efficacy result. Raw artifacts remain under the gitignored results tree.
set -euo pipefail
cd "$(dirname "$0")"                 # -> experiments/maze
REPO="$(cd ../.. && pwd)"
PY="$REPO/.venv/bin/python3"
BASE="$PWD/results/graph_persistent_dg/_exploratory_v7_dgwire"

if [[ ! -x "$PY" ]]; then
  echo "Python virtualenv not found: $PY" >&2
  exit 1
fi

if [[ "${1:-}" != "--analyze-only" ]]; then
  for ARM in "0.0:a00" "3.0:a30"; do
    A="${ARM%%:*}"
    TAG="${ARM##*:}"
    DIR="$BASE/$TAG"
    mkdir -p "$DIR"
    rm -f \
      "$DIR/s118.json" \
      "$DIR/s118.steps.json" \
      "$DIR/s118.steps.incremental.jsonl" \
      "$DIR/graph_seed118.json" \
      "$DIR/log.txt"
    INSIGHTSPIKE_DUMP_GRAPH="$DIR" PYTHONPATH="$REPO/src" \
    INSIGHTSPIKE_MIN_IMPORT=1 INSIGHTSPIKE_LITE_MODE=1 \
    "$PY" run_experiment_query.py \
      --maze-size 25 --max-steps 500 --seeds 1 --seed-start 118 \
      --max-hops 15 --sp-cand-topk 5 --curriculum-warmup-steps 500 --lambda-weight 0.01 \
      --vector-mode extended --sleep-propagate replay --sleep-guide off --steps-ultra-light \
      --link-radius 0.20 --action-policy softmax \
      --dg-action-alpha "$A" --dg-action-scale 10.0 \
      --step-log "$DIR/s118.steps.json" \
      --output "$DIR/s118.json" > "$DIR/log.txt" 2>&1
    echo "alpha=$A done -> $DIR"
  done
fi

"$PY" - "$BASE" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

base = Path(sys.argv[1])
arms = {"a00": 0.0, "a30": 3.0}
summary = {"experiment": "v7-dgwire-v1-source-query-action", "seed": 118, "arms": {}}
routes = {}
graph_bytes = {}

for tag, expected_alpha in arms.items():
    directory = base / tag
    result = json.loads((directory / "s118.json").read_text(encoding="utf-8"))
    graph_path = directory / "graph_seed118.json"
    graph_bytes[tag] = graph_path.read_bytes()
    graph = json.loads(graph_bytes[tag])
    step_path = directory / "s118.steps.json"
    if step_path.exists():
        rows = json.loads(step_path.read_text(encoding="utf-8"))
    else:
        step_path = directory / "s118.steps.incremental.jsonl"
        rows = [
            json.loads(line)
            for line in step_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    eval_rows = [row for row in rows if row.get("episode_phase") == "eval"]

    dg_config = result["config"]["graph_persistent_dg"]
    if float(dg_config["dg_action_alpha"]) != expected_alpha:
        raise SystemExit(f"{tag}: wrong alpha in result config")
    if float(dg_config["dg_action_scale"]) != 10.0:
        raise SystemExit(f"{tag}: wrong DG scale in result config")
    if dg_config.get("dg_projection") != "source-query-action":
        raise SystemExit(f"{tag}: wrong DG projection")
    if int(graph.get("schema_version", 0)) != 2:
        raise SystemExit(f"{tag}: graph dump schema mismatch")

    positive_edges = [
        edge for edge in graph.get("dg_edges", []) if float(edge[2]) > 0.0
    ]
    action_maps = [
        item for item in graph.get("dg_actions", []) if item[1]
    ]
    action_sizes = [
        float(size)
        for _node, mapping in action_maps
        for size in mapping.values()
        if float(size) > 0.0
    ]
    signal_by_position = {
        tuple(int(value) for value in node[:2]): {
            int(action): math.tanh(float(size) / 10.0)
            for action, size in mapping.items()
        }
        for node, mapping in action_maps
    }
    action_by_name = {"up": 0, "right": 1, "down": 2, "left": 3}
    opposite = {0: 2, 1: 3, 2: 0, 3: 1}
    policy_rows = []
    previous_row = None
    for row in eval_rows:
        position = tuple(int(value) for value in (row.get("position_pre") or [])[:2])
        signals = signal_by_position.get(position, {})
        feasible = [int(action) for action in (row.get("possible_moves") or [])]
        eligible = list(feasible)
        if (
            len(eligible) > 1
            and previous_row is not None
            and bool(previous_row.get("moved"))
        ):
            previous_action = action_by_name.get(str(previous_row.get("action")))
            masked = opposite.get(previous_action)
            filtered = [action for action in eligible if action != masked]
            if filtered:
                eligible = filtered
        feasible_values = [float(signals.get(action, 0.0)) for action in feasible]
        eligible_values = [float(signals.get(action, 0.0)) for action in eligible]
        selected_action = action_by_name.get(str(row.get("action")))
        selected_value = float(signals.get(selected_action, 0.0))
        signal_spread = (
            max(eligible_values) - min(eligible_values)
            if len(eligible_values) > 1
            else 0.0
        )
        policy_rows.append(
            {
                "feasible_positive": sum(value > 0.0 for value in feasible_values),
                "eligible_actions": len(eligible),
                "eligible_positive": sum(value > 0.0 for value in eligible_values),
                "eligible_max": max(eligible_values, default=0.0),
                "signal_spread": signal_spread,
                "selected_value": selected_value,
            }
        )
        previous_row = row

    feasible_steps = sum(item["feasible_positive"] > 0 for item in policy_rows)
    eligible_signal_steps = sum(item["eligible_positive"] > 0 for item in policy_rows)
    competitive_steps = sum(
        item["eligible_positive"] > 0
        and item["eligible_actions"] > 1
        and item["signal_spread"] > 1e-12
        for item in policy_rows
    )
    exposed_steps = eligible_signal_steps if expected_alpha > 0.0 else 0
    selected_values = [
        item["selected_value"]
        for item in policy_rows
        if item["selected_value"] > 0.0
    ]
    applied_steps = len(selected_values) if expected_alpha > 0.0 else 0
    selected_log_biases = [
        expected_alpha * value for value in selected_values
    ]
    routes[tag] = [
        (tuple(row.get("position_pre") or []), row.get("action"))
        for row in eval_rows
    ]
    run = next(
        run for run in result["runs"] if run.get("episode_phase") == "eval"
    )

    if not positive_edges or not action_sizes:
        raise SystemExit(f"{tag}: DG annotation produced no positive signal")
    if not eval_rows:
        raise SystemExit(f"{tag}: missing eval telemetry")
    if expected_alpha > 0.0 and (exposed_steps == 0 or applied_steps == 0):
        raise SystemExit(f"{tag}: DG readout wiring did not engage")

    summary["arms"][tag] = {
        "alpha": expected_alpha,
        "success": bool(run["success"]),
        "steps": int(run["steps"]),
        "dead_end_steps": int(run["dead_end_steps"]),
        "graph_nodes": len(graph["nodes"]),
        "graph_edges": len(graph["edges"]),
        "dg_positive_edges": len(positive_edges),
        "dg_source_query_nodes": len(action_maps),
        "dg_state_actions": len(action_sizes),
        "dg_size_max": max(action_sizes),
        "dg_feasible_signal_steps": feasible_steps,
        "dg_eligible_signal_steps": eligible_signal_steps,
        "dg_candidate_value_max": max(
            (item["eligible_max"] for item in policy_rows),
            default=0.0,
        ),
        "dg_exposed_steps": exposed_steps,
        "dg_competitive_steps": competitive_steps if expected_alpha > 0.0 else 0,
        "dg_applied_steps": applied_steps,
        "dg_selected_value_max": max(selected_values, default=0.0),
        "dg_selected_value_mean": (
            sum(selected_values) / len(selected_values) if selected_values else 0.0
        ),
        "dg_selected_log_bias_max": max(selected_log_biases, default=0.0),
    }

route_a = routes["a00"]
route_b = routes["a30"]
if graph_bytes["a00"] != graph_bytes["a30"]:
    raise SystemExit("sleep graph differs between control and active arms")
first_divergence = next(
    (
        index
        for index, pair in enumerate(zip(route_a, route_b))
        if pair[0] != pair[1]
    ),
    None,
)
if first_divergence is None and len(route_a) != len(route_b):
    first_divergence = min(len(route_a), len(route_b))
summary["comparison"] = {
    "sleep_graph_identical": True,
    "route_identical": route_a == route_b,
    "first_route_divergence_step": first_divergence,
    "delta_steps_a30_minus_a00": (
        summary["arms"]["a30"]["steps"] - summary["arms"]["a00"]["steps"]
    ),
    "delta_dead_end_steps_a30_minus_a00": (
        summary["arms"]["a30"]["dead_end_steps"]
        - summary["arms"]["a00"]["dead_end_steps"]
    ),
    "wiring_manipulation_check": "PASS",
    "efficacy_manipulation_check": (
        "PASS"
        if summary["arms"]["a30"]["dg_competitive_steps"] > 0
        else "INCONCLUSIVE_NO_COMPETITIVE_EXPOSURE"
    ),
    "scope": "exploratory-single-seed",
}

(base / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "BOTH DONE: manipulation summary -> $BASE/summary.json"
