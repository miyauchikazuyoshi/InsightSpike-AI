from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


Position = Tuple[int, int]
Step = Dict[str, Any]


def _iter_steps(steps: Iterable[Dict[str, Any]]) -> Iterable[Step]:
    for rec in steps:
        if isinstance(rec, dict):
            yield rec


def extract_exploration_metrics(steps: List[Step], maze_size: int) -> Dict[str, Any]:
    """
    Compute maze exploration efficiency metrics from step logs.

    Returns a dict with exploration_ratio, avg_visits_per_cell, and counts.
    The function is resilient to partially missing fields.
    """
    visited: Dict[Position, int] = {}
    total_steps = 0
    for rec in _iter_steps(steps):
        pos = rec.get("position")
        if isinstance(pos, list) and len(pos) >= 2:
            key = (int(pos[0]), int(pos[1]))
            visited[key] = visited.get(key, 0) + 1
        total_steps += 1

    unique_cells = len(visited)
    total_cells = int(maze_size) * int(maze_size) if int(maze_size) > 0 else 0
    exploration_ratio = (unique_cells / total_cells) if total_cells > 0 else 0.0
    avg_visits_per_cell = (sum(visited.values()) / unique_cells) if unique_cells > 0 else 0.0

    return {
        "exploration_ratio": float(exploration_ratio),
        "avg_visits_per_cell": float(avg_visits_per_cell),
        "unique_cells_visited": int(unique_cells),
        "total_cells": int(total_cells),
        "total_steps": int(total_steps),
    }


def detect_backtracks(steps: List[Step]) -> List[Dict[str, Any]]:
    """
    Detect backtrack segments using AG/DG events as anchors.

    A segment starts when we detect a dead-end entry and AG fires (attention on stall),
    and ends when DG fires (decision made / route confirmed). If explicit 'hit_wall' is
    not available, we approximate dead-end by possible_moves <= 1.
    """
    segments: List[Dict[str, Any]] = []
    in_deadend = False
    deadend_entry_idx = None
    ag_start_idx = None

    for idx, rec in enumerate(_iter_steps(steps)):
        possible_moves = rec.get("possible_moves")
        if isinstance(possible_moves, list):
            try:
                is_deadend_now = (len(possible_moves) <= 1)
            except Exception:
                is_deadend_now = False
        else:
            is_deadend_now = bool(rec.get("is_dead_end", False))

        if not in_deadend and is_deadend_now:
            in_deadend = True
            deadend_entry_idx = idx

        if rec.get("ag_fire", False):
            # Start of a backtrack response
            if in_deadend and ag_start_idx is None:
                ag_start_idx = idx

        if rec.get("dg_fire", False) and ag_start_idx is not None:
            # End of backtrack; DG confirms a route
            detection_delay = 0 if deadend_entry_idx is None else max(0, ag_start_idx - deadend_entry_idx)
            length = max(0, idx - ag_start_idx)
            segments.append({
                "start": int(ag_start_idx),
                "end": int(idx),
                "length": int(length),
                "detection_delay": int(detection_delay),
            })
            # Reset state for next segment
            in_deadend = False
            deadend_entry_idx = None
            ag_start_idx = None

        # If not dead-end anymore, reset entry if no active AG
        if in_deadend and not is_deadend_now and ag_start_idx is None:
            in_deadend = False
            deadend_entry_idx = None

    return segments


def summarize_backtrack(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not segments:
        return {
            "n_backtracks": 0,
            "avg_backtrack_length": 0.0,
            "deadend_detection_delay": 0.0,
            "total_backtrack_steps": 0,
        }
    n = len(segments)
    total_len = sum(int(s.get("length", 0)) for s in segments)
    total_delay = sum(int(s.get("detection_delay", 0)) for s in segments)
    return {
        "n_backtracks": int(n),
        "avg_backtrack_length": float(total_len / n),
        "deadend_detection_delay": float(total_delay / n),
        "total_backtrack_steps": int(total_len),
    }


def summarize_gates(steps: List[Step]) -> Dict[str, Any]:
    ag = 0
    dg = 0
    n = 0
    for rec in _iter_steps(steps):
        n += 1
        if rec.get("ag_fire", False):
            ag += 1
        if rec.get("dg_fire", False):
            dg += 1
    return {
        "ag_fire_count": int(ag),
        "dg_fire_count": int(dg),
        "ag_fire_rate": float(ag / n) if n else 0.0,
        "dg_fire_rate": float(dg / n) if n else 0.0,
        "ag_dg_ratio": float(dg / ag) if ag else 0.0,
    }

