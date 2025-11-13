---
layout: page
title: Phase‑1 PoC — Maze & RAG
permalink: /phase1/
---

# Phase‑1 PoC — Maze & RAG

## Maze (partial observability)

- Single run (seed0):
  ```bash
  python experiments/maze-query-hub-prototype/run_experiment_query.py \
    --preset paper --maze-size 25 --max-steps 300 \
    --output tmp/seed0_summary.json --step-log tmp/seed0_steps.json
  ```
- 60‑seed batch (25×25, s500) + table update:
  ```bash
  python scripts/run_maze_batch_and_update.py --mode l3 --seeds 60 --workers 4 --update-tex
  ```
- Aggregates: `docs/paper/data/maze_25x25_l3_s500.json`

<p align="center">
  <img alt="Maze demo (seed0 short)" src="../docs/images/maze_seed0_recon.gif" width="560" />
  <br/>
  <em>Seed0 (short). See the repo for the interactive HTML viewer.</em>
  
</p>

## RAG (equal‑resources)

- See paper figures for PSZ/operating curves and the aggregation protocol
- Use `make` targets / scripts to export PSZ tables/figures (see docs/paper/generate_* scripts)

<!-- BASELINE:BEGIN -->

### Baseline Comparison (Static RAG vs geDIG)

| Metric | Static RAG | geDIG |
|---|---:|---:|
| Acceptance | 0.360 | 0.380 |
| FMR | 0.640 | 0.620 |
| Latency P50 | 160 ms | 200 ms |

<!-- BASELINE:END -->
