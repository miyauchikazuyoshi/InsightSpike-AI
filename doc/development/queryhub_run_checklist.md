# Maze Query‑Hub: Run Checklist (ΔIG sign + logging)

This note captures the guardrails to keep query‑hub runs consistent with the paper spec and to avoid missing fields during logging.

Recommended defaults (set by the driver already):

- MAZE_GEDIG_IG_NONNEG=0  (do not clamp IG ≥ 0)
- MAZE_GEDIG_IG_DELTA=after_before  (ΔH = H_after − H_before; “秩序化で負/正の両方を許容”)

Driver improvements in this revision:

- StepRecord: duplicate fields removed (query_node_post), duplicate JSON keys cleaned.
- Episode summary now records series for aggregation:
  - delta_sp_series, delta_sp_min_series
  - eval_time_ms_series
  - multihop_best_hop
- Aggregation now returns:
  - success_rate, avg_steps, avg_edges, g0_mean, gmin_mean, avg_k_star
  - avg_delta_sp, avg_delta_sp_min
  - best_hop_mean and histogram buckets (0/1/2/3+)
  - avg_time_ms_eval, p95_time_ms_eval

Minimal run examples:

```
python experiments/maze-query-hub-prototype/run_experiment_query.py \
  --maze-size 25 --max-steps 180 --max-hops 3 \
  --lambda-weight 1.0 --sp-beta 0.2 \
  --theta-ag 0.0 --theta-dg 0.0 \
  --linkset-mode --norm-base link \
  --output experiments/maze-query-hub-prototype/results/run_25_summary.json \
  --step-log experiments/maze-query-hub-prototype/results/run_25_steps.json

python experiments/maze-query-hub-prototype/build_reports.py \
  --summary experiments/maze-query-hub-prototype/results/run_25_summary.json \
  --steps   experiments/maze-query-hub-prototype/results/run_25_steps.json \
  --out     experiments/maze-query-hub-prototype/results/run_25_interactive.html
```

Notes:

- ΔIG = ΔH + γ·ΔSP が既定。ΔH は after−before。
- GED normalization uses candidate‑base by default:
  Cmax = c_node + |S_link|·c_edge （|S_link|=Top‑Lのリンク本数; |S_link|=0のときは最小1相当で分母ゼロ回避）
  hop0/≥1で同じスキーム（CLI `--norm-base link` が既定）
- SP gain uses fixed‑before‑pairs when selected, with optional DS reuse.
- If you need the alternative orientation (ΔH=before−after), export `MAZE_GEDIG_IG_DELTA=before_after` before running.
