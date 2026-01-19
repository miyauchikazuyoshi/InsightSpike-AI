# Maze Navigation (15x15)

## Claim
- geDIG gating keeps success rate high in partial-observation mazes (15x15 ~98% in the paper setting).

## Setup
- Python 3.10+
- `poetry install`
- No external datasets (mazes are generated on the fly).

## Run
```bash
make reproduce-maze15
```

## Metrics
- `success_rate`, `avg_steps`, `avg_edges`, `g0_mean`, `gmin_mean`
- Gate fire rates (`ag_rate`, `dg_rate`) and compression metrics

## Ablation
See `experiments/maze-query-hub-prototype/ABLATION_PLAN.md` for detailed variants.

Full-scale run (15x15, max_steps=250, seeds=12, max_hops=3, preset=paper).

| Variant | success_rate | avg_steps | gmin_mean | ag_rate | dg_rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| Full geDIG (AG+DG+F) | 1.00 | 92.7 | -0.3131 | 0.057 | 0.967 | baseline |
| AG only | 1.00 | 92.7 | -0.3131 | 0.057 | 0.967 | `--dg-commit-policy always` |
| DG only | 1.00 | 92.7 | -0.5829 | 1.000 | 0.967 | `--theta-ag -1.0` |
| F without ΔSP | 1.00 | 92.7 | -0.3131 | 0.057 | 0.967 | `--sp-beta 0` (ΔSP=0 here) |

## Expected outputs
- `experiments/maze-query-hub-prototype/results/l3_only/*_summary.json`
- `experiments/maze-query-hub-prototype/results/l3_only/*_steps.json`
- `docs/paper/data/maze_15x15_l3_s250.json`
- `docs/paper/data/maze_15x15_l3_s250.csv`
- `docs/paper/data/maze_15x15_ablation_fullscale.json`

## Known limitations
- Runtime grows with `seeds` and `max-steps`.
- Results vary with random seeds and preset knobs.
