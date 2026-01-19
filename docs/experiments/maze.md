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

## Expected outputs
- `experiments/maze-query-hub-prototype/results/l3_only/*_summary.json`
- `experiments/maze-query-hub-prototype/results/l3_only/*_steps.json`
- `docs/paper/data/maze_15x15_l3_s250.json`
- `docs/paper/data/maze_15x15_l3_s250.csv`

## Known limitations
- Runtime grows with `seeds` and `max-steps`.
- Results vary with random seeds and preset knobs.
