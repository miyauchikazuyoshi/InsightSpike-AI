# Maze Navigator: Current Spec and Controls

This document summarizes the current behavior, knobs, and workflow of the maze navigation experiment stack in this repository.

## Overview

- Core loop: MazeNavigator.step()
  - Observe episodes (current cell + four directions) and add every episode as a node
  - Build a query vector at the current position (prefer_unexplored=True)
  - Wire graph:
    - L1 candidate search (all memory; see L1 section)
    - Optional forced L1 wiring for frontier-like close neighbors
    - geDIG-based wiring (optionally using hop=min decision)
  - Capture metrics (geDIG, structural deltas, optional SP delta)
  - Select next action (DecisionEngine)

- Graph evaluates “geDIG = ΔGED_norm − λ·IG” like score; positive ≈ expansion, negative ≈ shortcut/loop. NA/DA two-stage gating is implemented at the evaluator/aggregator level (see NA/DA).

- Visualization: docs/images/gedegkaisetsu/threshold_dfs_live/index.html
  - Map + path playback
  - geDIG/ΔSP chart, growth chart (nodes/edges)
  - Episode graph snapshot with wiring color-coded (geDIG/forced-L1/query/trajectory/etc.)

## L1 Search and Wiring

- Candidates:
  - Time window removed. The candidate set is all earlier episodes (positioned nodes); walls are included as candidates for recall.
  - Optional spatial gate is disabled by default (`MAZE_SPATIAL_GATE=0`).

- Distance:
  - Weighted L2 with optional unit-norm. Default weights:
    - Position: (1,1)
    - Direction: (0,0)
    - Wall: 6
    - Visit(log1p): 4
    - Success/goal: (0,0)
  - Typical thresholds: `MAZE_L1_NORM_TAU=1.05..1.2` with `MAZE_L1_CAND_TOPK=6..10`.

- Forced-L1 wiring (frontier-like only):
  - Enable: `MAZE_WIRING_FORCE_L1=1`
  - Threshold: `MAZE_WIRING_FORCE_L1_TAU` (e.g., 0.6–0.8)
  - Limit: `MAZE_WIRING_FORCE_L1_TOPK` (e.g., 2)
  - Frontier-like heuristic: either the target episode has `visit_count==0` or that position has any non-wall `visit_count==0` episode. Such close neighbors are wired immediately before geDIG evaluation. Walls are not specially filtered here; heavy weights on wall/visit keep them naturally disfavored.

- geDIG wiring:
  - Decision score can use `hop=min` (multi-hop) via `MAZE_USE_HOP_DECISION=1` and `MAZE_HOP_DECISION_LEVEL=min`.
  - Acceptance threshold: `MAZE_GEDIG_THRESHOLD` with `MAZE_WIRING_TOPK` and optional `MAZE_WIRING_MIN_ACCEPT=1`.

## NA/DA Gating and BT

- NA (0-hop): threshold `MAZE_NA_GE_THRESH` (default ≈ −0.005). NA is active when 0-hop geDIG is strictly above this value.
- DA aggregation for BT (`bt_eval_value`):
  - `MAZE_BT_AGG=na_min` → at NA steps, `bt_eval = min(0hop, min-hop)`; when NA is not active, DA is not used and `bt_eval = 0hop`.
- BT trigger (pure geDIG gate): fires only at NA steps and only when `bt_eval_value <= MAZE_BACKTRACK_THRESHOLD` in that same frame.

## SP (Average Shortest Path)

- Optional global SP (position-level contracted graph) with sampling:
  - Enable: `MAZE_SP_GLOBAL_ENABLE=1`, `MAZE_SP_POSLEVEL=1`
  - Control: `MAZE_SP_GLOBAL_SAMPLES`, `MAZE_SP_GLOBAL_BUDGET_MS`
  - NA-forced SP: `MAZE_SP_FORCE_ON_NA=1` with `MAZE_SP_FORCE_SAMPLES/BUDGET_MS`
- ΔSP is after−before; negative means “closer” (shortcut/loop benefit).

## Query Hub (QHUB)

- Ephemeral query node connected to the current episode and L1-selected candidates used for recall/analysis without polluting wiring:
  - `MAZE_USE_QUERY_HUB=1`, `MAZE_QUERY_HUB_PERSIST=0`, `MAZE_QUERY_HUB_CONNECT_CURRENT=1`.

## Viewer (threshold_dfs_live)

- Toggles: geDIG chart, ΔSP, growth (nodes/edges), episode graph, edge-layer filters (geDIG/query/traj/hub/spatial/other), width=|geDIG|, dark background.
- Edge colors: geDIG=green, forced-L1=magenta, query=blue, trajectory=dark gray, hub=gray, spatial=orange, other=blue‑violet.

## High-signal presets

- 15×15 quick inspect
  - `MAZE_USE_QUERY_HUB=1`
  - L1: weights=(1,1,0,0,6,4,0,0), τ=1.05, CAND_TOPK=8
  - Forced-L1: ON, τ=0.7, TOPK=2
  - Wiring: TOPK=3, MIN_ACCEPT=1, hop=min
  - SP: poslevel ON, SAMPLES=200, BUDGET=15ms, NA-force SP ON

- 25×25 lightweight
  - Same as above; reduce SP SAMPLES/BUDGET (e.g., 120/8) and max-steps≈140–180 to avoid timeouts

## Notes

- Spatial candidate gates are OFF by default: `SPATIAL_GATE=0`, and the previous NA-only ring expansion is also OFF by default (`ESCALATE_RING=0`).
- Time-windowed L1 has been removed in favor of a global recall (all memory) to encourage pulling distant branches.

## Housekeeping

- Stray experiment directories from the repository root have been moved to:
  - `experiments/maze-navigation-enhanced/results/root_scatter_<timestamp>/`
  - Examples of moved entries included `--max-steps`, `--seed`, numeric folders (`25/42/600`), and ad‑hoc env‑named folders.
