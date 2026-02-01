#!/usr/bin/env python3
from __future__ import annotations

"""
Export event schema, event stats, and auto-calibrated event weights from maze step logs.

Usage:
  .venv/bin/python experiments/maze-query-hub-prototype/tools/export_event_weights.py \
    --steps results/maze-local/steps.json \
    --summary results/maze-local/summary.json \
    --out-dir results/maze-local/sleep_events
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


EVENT_SCHEMA: Dict[str, Dict[str, Any]] = {
    "goal_reached": {
        "polarity": "positive",
        "magnitude": "large",
        "default_weight": 2.0,
        "description": "Reached the goal state (terminal).",
    },
    "blocked": {
        "polarity": "negative",
        "magnitude": "large",
        "default_weight": -2.0,
        "description": "Attempted an invalid move (wall/blocked).",
    },
    "timeout": {
        "polarity": "negative",
        "magnitude": "medium",
        "default_weight": -1.0,
        "description": "Episode ended without success (step cap).",
    },
    "deadend": {
        "polarity": "negative",
        "magnitude": "medium",
        "default_weight": -0.7,
        "description": "Entered a dead-end region.",
    },
    "stuck": {
        "polarity": "negative",
        "magnitude": "medium",
        "default_weight": -0.6,
        "description": "Local cycling / stuckness detected.",
    },
    "revisit": {
        "polarity": "negative",
        "magnitude": "small",
        "default_weight": -0.3,
        "description": "Visited a previously seen cell.",
    },
    "immediate_backtrack": {
        "polarity": "negative",
        "magnitude": "small",
        "default_weight": -0.2,
        "description": "Returned to the previous cell immediately.",
    },
    "move_success": {
        "polarity": "positive",
        "magnitude": "small",
        "default_weight": 0.2,
        "description": "Move executed successfully.",
    },
    "novel_cell": {
        "polarity": "positive",
        "magnitude": "small",
        "default_weight": 0.3,
        "description": "Entered a new cell for the first time.",
    },
    "progress": {
        "polarity": "positive",
        "magnitude": "small",
        "default_weight": 0.3,
        "description": "Moved along a higher-value direction (Sleep Q/plan).",
    },
}


ACTION_TO_ID = {
    "up": 0,
    "right": 1,
    "down": 2,
    "left": 3,
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_steps(path: Path) -> List[Dict[str, Any]]:
    data = _load_json(path)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("steps"), list):
            return [x for x in data["steps"] if isinstance(x, dict)]
        out: List[Dict[str, Any]] = []
        for v in data.values():
            if isinstance(v, list):
                out.extend([x for x in v if isinstance(x, dict)])
        return out
    return []


def _pos(value: Any) -> Optional[Tuple[int, int]]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except Exception:
            return None
    return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _success_map(summary: Dict[str, Any]) -> Dict[Tuple[int, str], bool]:
    out: Dict[Tuple[int, str], bool] = {}
    curriculum = summary.get("curriculum") if isinstance(summary, dict) else None
    if isinstance(curriculum, dict):
        per_seed = curriculum.get("per_seed") or {}
        if isinstance(per_seed, dict):
            for seed_key, entry in per_seed.items():
                try:
                    seed = int(seed_key)
                except Exception:
                    continue
                warmup = entry.get("warmup") or {}
                evalr = entry.get("eval") or {}
                if isinstance(warmup, dict):
                    out[(seed, "warmup")] = bool(warmup.get("success", False))
                if isinstance(evalr, dict):
                    out[(seed, "eval")] = bool(evalr.get("success", False))
    runs = summary.get("runs") or []
    if isinstance(runs, list):
        for row in runs:
            if not isinstance(row, dict):
                continue
            seed = row.get("seed")
            if seed is None:
                continue
            phase = str(row.get("episode_phase", "main"))
            out[(int(seed), phase)] = bool(row.get("success", False))
    return out


def _goal_map(summary: Dict[str, Any]) -> Dict[int, Tuple[int, int]]:
    out: Dict[int, Tuple[int, int]] = {}
    maze_data = summary.get("maze_data") or {}
    if isinstance(maze_data, dict):
        for seed_key, snap in maze_data.items():
            if not isinstance(snap, dict):
                continue
            try:
                seed = int(seed_key)
            except Exception:
                continue
            goal = snap.get("goal_pos")
            gp = _pos(goal)
            if gp is not None:
                out[seed] = gp
    return out


def _detect_events(
    row: Dict[str, Any],
    state: Dict[str, Any],
    goal_pos: Optional[Tuple[int, int]],
) -> List[str]:
    events: List[str] = []
    pos_pre = _pos(row.get("position_pre") or row.get("position"))
    pos_post = _pos(row.get("position_post") or row.get("position"))
    moved = bool(row.get("moved")) if "moved" in row else (pos_pre is not None and pos_post is not None and pos_pre != pos_post)
    if moved:
        events.append("move_success")
    if pos_post is not None:
        visits = state["visits"].get(pos_post, 0) + 1
        state["visits"][pos_post] = visits
        if visits == 1:
            events.append("novel_cell")
        elif visits >= 2:
            events.append("revisit")
    action = row.get("action")
    action_id = ACTION_TO_ID.get(str(action))
    possible_moves = row.get("possible_moves")
    if isinstance(possible_moves, list) and action_id is not None:
        if action_id not in [int(x) for x in possible_moves if isinstance(x, (int, float))]:
            events.append("blocked")
    if not moved and pos_pre is not None and pos_post is not None and pos_pre == pos_post:
        if "blocked" not in events:
            events.append("blocked")
    if bool(row.get("is_dead_end", False)):
        events.append("deadend")
    cortisol_reason = str(row.get("cortisol_reason", ""))
    cortisol_stuck = _safe_int(row.get("cortisol_stuck_streak", 0))
    if "stuck" in cortisol_reason or cortisol_stuck > 0:
        events.append("stuck")
    if state.get("prev_pos_pre") is not None and pos_post is not None:
        if pos_post == state["prev_pos_pre"]:
            events.append("immediate_backtrack")
    if goal_pos is not None and pos_post is not None and pos_post == goal_pos:
        events.append("goal_reached")
    if _safe_float(row.get("sleep_q_adv", 0.0)) > 0.0 or bool(row.get("sleep_guided", False)):
        events.append("progress")
    state["prev_pos_pre"] = pos_pre
    return events


def _log_odds_weight(p_s: float, p_f: float, w_min: float, w_max: float) -> float:
    w = math.log(max(p_s, 1e-12) / max(p_f, 1e-12))
    return max(w_min, min(w_max, w))


def main() -> None:
    ap = argparse.ArgumentParser(description="Export event schema/stats/weights from step logs")
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epsilon", type=float, default=1e-6)
    ap.add_argument("--w-min", type=float, default=-3.0)
    ap.add_argument("--w-max", type=float, default=3.0)
    args = ap.parse_args()

    steps = _load_steps(args.steps)
    summary: Dict[str, Any] = {}
    success_map: Dict[Tuple[int, str], bool] = {}
    goal_map: Dict[int, Tuple[int, int]] = {}
    if args.summary is not None and args.summary.exists():
        summary = _load_json(args.summary)
        if isinstance(summary, dict):
            success_map = _success_map(summary)
            goal_map = _goal_map(summary)

    steps.sort(key=lambda r: (int(r.get("seed", 0)), str(r.get("episode_phase", "main")), int(r.get("step", 0))))
    per_episode: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for row in steps:
        seed = _safe_int(row.get("seed", -1), -1)
        if seed < 0:
            continue
        phase = str(row.get("episode_phase", "main"))
        per_episode.setdefault((seed, phase), []).append(row)

    stats: Dict[str, Dict[str, int]] = {name: {"success": 0, "fail": 0} for name in EVENT_SCHEMA.keys()}
    totals = {"success_steps": 0, "fail_steps": 0, "episodes": len(per_episode)}

    for (seed, phase), rows in per_episode.items():
        success = success_map.get((seed, phase))
        if success is None:
            success = True
        goal_pos = goal_map.get(seed)
        state = {"visits": {}, "prev_pos_pre": None}
        for row in rows:
            events = _detect_events(row, state, goal_pos)
            for ev in events:
                if ev not in stats:
                    continue
                if success:
                    stats[ev]["success"] += 1
                else:
                    stats[ev]["fail"] += 1
            if success:
                totals["success_steps"] += 1
            else:
                totals["fail_steps"] += 1
        if not success:
            stats["timeout"]["fail"] += 1

    weights: Dict[str, float] = {}
    for name, meta in EVENT_SCHEMA.items():
        default_w = float(meta.get("default_weight", 0.0))
        if totals["success_steps"] == 0 or totals["fail_steps"] == 0:
            weights[name] = default_w
            continue
        p_s = (stats[name]["success"] + args.epsilon) / (totals["success_steps"] + args.epsilon * 2.0)
        p_f = (stats[name]["fail"] + args.epsilon) / (totals["fail_steps"] + args.epsilon * 2.0)
        weights[name] = _log_odds_weight(p_s, p_f, args.w_min, args.w_max)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "event_schema.json").write_text(json.dumps({"events": EVENT_SCHEMA}, indent=2), encoding="utf-8")
    (args.out_dir / "event_stats.json").write_text(json.dumps({"totals": totals, "events": stats}, indent=2), encoding="utf-8")
    (args.out_dir / "event_weights.json").write_text(
        json.dumps(
            {
                "weights": weights,
                "meta": {
                    "epsilon": args.epsilon,
                    "w_min": args.w_min,
                    "w_max": args.w_max,
                    "success_steps": totals["success_steps"],
                    "fail_steps": totals["fail_steps"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"out_dir": str(args.out_dir), "events": list(EVENT_SCHEMA.keys())}, indent=2))


if __name__ == "__main__":
    main()
