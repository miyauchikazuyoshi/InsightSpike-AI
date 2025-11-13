---
layout: page
title: Tutorial — Trace a Spike Decision
permalink: /tutorials/trace/
---

# Tutorial — Trace a Spike Decision

This short tutorial walks through a single query in the maze PoC and inspects
how geDIG evaluates \(\mathcal{F}\) and gates (AG/DG).

## 1) Run a short episode

```bash
python experiments/maze-query-hub-prototype/run_experiment_query.py \
  --preset paper --maze-size 25 --max-steps 60 \
  --output tmp/trace_summary.json --step-log tmp/trace_steps.json
```

## 2) Inspect a step record

Open `tmp/trace_steps.json` and look for fields:

- `g0`, `gmin`: per‑hop gauge values (0‑hop and best multi‑hop)
- `ag_fire`, `dg_fire`: whether AG/DG fired at this step
- `selected_links`: candidate link adopted at hop0 (for evaluation)
- `linkset_delta_*`: per‑step ΔGED/ΔH/ΔSP used to compute \(\mathcal{F}\)

## 3) Interpret the decision

- If `g0` is high (ambiguous), AG triggers retrieval
- If some `g(h)` is low enough, DG confirms and commits (Accept)
- Otherwise the system rejects/holds (no commit) and continues exploration

See the paper for formal definitions and the PSZ/SLO criteria.

