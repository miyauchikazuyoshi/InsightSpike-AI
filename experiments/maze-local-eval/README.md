# Maze Local Evaluation (25×25)

This experiment exercises the localized geDIG pipeline after the structural-cost refactor. It replays 25×25 random mazes and records per-step geDIG scores together with HTML visualizations for manual inspection.

## How to Run

```bash
cd experiments/maze-local-eval
python run_experiment.py \
  --size 25 \
  --seeds 20 \
  --feature-profile option_b \
  --theta-na 0.35 \
  --theta-bt 0.30 \
  --output results/25x25/local_eval_option_b_raw.json \
  --summary results/25x25/local_eval_option_b_summary.csv \
  --log-steps results/25x25/step_logs/seed_{seed}.csv
```

The raw JSON contains the full path trace, hop-wise geDIG values, and event logs for each seed. Re-run the visualization generator to refresh the HTML dashboards:

```bash
python ../maze-online-phase1-querylog/visualization/generate_report.py \
  --input results/25x25/local_eval_option_b_raw.json \
  --output visualization/reports_25x25_option_b
```

## g₀ Evaluation Flow

1. **Observation set.** At every step the navigator gathers the four neighbourhood directions. Traversable cells, previously visited passages, and walls are all encoded as candidates so that disagreement with the query vector shows up in the IG term even if the agent cannot advance.
2. **Virtual wiring.** Observed candidates are treated as fully wired when evaluating the 0-hop graph. This matches the paper’s “query + connected nodes” definition: the hypothetical neighbourhood graph has all four spokes present regardless of whether the agent has already traversed them.
3. **Structural cost.** The adapter calls `normalized_ged` on the local before/after neighbourhood graphs. The value is always ≥ 0; larger means the hypothetical wiring introduces more edit operations. We surface that directly as `structural_cost`.
4. **Information gain.** Each candidate is represented by an 8-dim feature vector `(x/W, y/H, dx, dy, wall, log(1+visits), success, goal)`. Walls therefore contribute to the entropy calculation. `ΔIG = H_before - H_after` is normalized by `log 4`.
5. **g₀ / g_min.** For hop 0 the geDIG score is `g₀ = structural_cost - λ·ΔIG` with `λ = 0.5`. Multi-hop rolls this forward with a radius of 3 (decay 0.7) to obtain `g_min`.
6. **Thresholds.** The backtrack trigger now compares `g₀` against positive costs: `θ_BT = 0.30`, while the NA (no-action) gate uses `θ_NA = 0.35`. Higher values are “worse”, so both thresholds are upper bounds.

## Outputs

| Artifact | Path | Notes |
| --- | --- | --- |
| Raw per-seed logs | `results/25x25/local_eval_option_b_raw.json` | Full run data including per-step structural cost, IG, and event history. |
| Summary CSV | `results/25x25/local_eval_option_b_summary.csv` | Aggregate metrics across all seeds (success rate, mean g₀, etc.). |
| HTML dashboards | `visualization/reports_25x25_option_b/` | Per-seed canvases with timeline plots for `g₀`, `g_min`, ΔSP, and threshold overlays. |

To inspect a seed, open `visualization/reports_25x25_option_b/seed_{n}/index.html` in a browser. The timeline now highlights positive thresholds, and the event log shows both `structural_cost` and `g₀` for each action.

### 15×15 quick check

同じ設定で 15×15 迷路（seed=0–19）を走らせた結果:

| Profile | g₀ mean | g_min mean | NA rate | Output |
| --- | --- | --- | --- | --- |
| default | 0.6889 | 0.6887 | 0.1220 | `results/15x15/local_eval_default_*`, `visualization/reports_15x15_default/` |
| option_a | 0.6937 | 0.6935 | 0.1245 | `results/15x15/local_eval_option_a_*`, `visualization/reports_15x15_option_a/` |
| option_b | 0.6963 | 0.6963 | 0.1232 | `results/15x15/local_eval_option_b_*`, `visualization/reports_15x15_option_b/` |

いずれも成功率 100%、`θ_BT=0.30`/`θ_NA=0.35` で安定しており、g₀ は 0.65±0.02 付近に集中しています。

## Follow-up

- Rerun the RAG experiments once the reader pipeline is updated to consume `structural_cost`.
- Remove the deprecated `structural_improvement` field after downstream consumers migrate.
- Extend the HTML to surface σ-normalised IG once the paper revision finalizes the formula.
