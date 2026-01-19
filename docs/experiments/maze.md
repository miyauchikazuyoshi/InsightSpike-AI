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

## Ablation (planned)
See `experiments/maze-query-hub-prototype/ABLATION_PLAN.md` for detailed variants.

| Variant | Intent | Status |
|---|---|---|
| Full geDIG (AG+DG+F) | Paper preset default | baseline |
| AG only | Disable DG commits (`--dg-commit-policy never`) | TODO |
| DG only | Suppress AG expansion (set `--theta-ag` high) | TODO |
| F without ΔSP | Remove SP contribution (`--sp-beta 0`) | TODO |

## Expected outputs
- `experiments/maze-query-hub-prototype/results/l3_only/*_summary.json`
- `experiments/maze-query-hub-prototype/results/l3_only/*_steps.json`
- `docs/paper/data/maze_15x15_l3_s250.json`
- `docs/paper/data/maze_15x15_l3_s250.csv`

## Known limitations
- Runtime grows with `seeds` and `max-steps`.
- Results vary with random seeds and preset knobs.
