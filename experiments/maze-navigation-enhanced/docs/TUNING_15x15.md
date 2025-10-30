15x15 Tuning Guide
===================

This guide describes how to grid‑search key parameters for the 15×15 maze and pick a robust configuration before running unified experiments.

Quick Run
---------

```
PYTHONPATH=experiments/maze-navigation-enhanced/src \
python experiments/maze-navigation-enhanced/src/analysis/tune_15x15.py \
  --seeds 12 --fast --grid-small
```

- success_rate: fraction of seeds reaching goal within the step budget
- ratio_vs_bfs: avg steps (moves) vs BFS shortest (smaller is better)
- backtrack_step_rate: backtrack steps / total steps on successful runs
- avg_edges: average graph edges (structure compactness proxy)

Top‑5 configurations are printed and the full JSON report is saved under:
`experiments/maze-navigation-enhanced/results/tuning_15x15/`.

Recommended Toggles (stable defaults)
-------------------------------------

Environment variables applied by the tuner (can be overridden):
- `NAV_DISABLE_DIAMETER=1` to reduce measurement overhead
- `MAZE_BT_PLAN_FREEZE=1`, `MAZE_BT_REPLAN_STUCK_N=2`, `MAZE_BACKTRACK_COOLDOWN=80`
- `MAZE_BT_DYNAMIC=1` (stagnation‑aware backtracking)
- `MAZE_GEDIG_LOCAL_NORM=1`, `MAZE_GEDIG_SP_GAIN=1`

Use `--fast` to trim instrumentation while maintaining comparable decisions.

Larger Grid
-----------

```
PYTHONPATH=experiments/maze-navigation-enhanced/src \
python experiments/maze-navigation-enhanced/src/analysis/tune_15x15.py \
  --seeds 24 --topk 3 4 --gedig-th -0.20 -0.18 -0.15 -0.12 \
  --bt-th -0.30 -0.25 -0.22 -0.18
```

Apply Best Config
-----------------

1) Pick a top configuration from the report (e.g., `topk3_g-0.18_bt-0.22`).
2) Update `experiments/maze-navigation-enhanced/configs/15x15.yaml`:

```
maze:
  size: 15
  seeds: 20
  max_steps_factor: 3.0

gedig:
  threshold: -0.18
  backtrack_threshold: -0.22

env:
  MAZE_L1_CAND_TOPK: 3
```

3) For unified experiments, prefer consuming values via the preset loader:

```
from utils.preset_loader import load_preset
cfg = load_preset(preset_name='15x15')
```

Next Steps
----------

- Refactor runners (`clean_maze_run.py`) to read thresholds/top‑k from the preset and produce a standard summary schema
- Add a minimal comparison driver (geDIG vs simple) using the same preset, then scale to 25×25 once stable

