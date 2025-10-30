# Baseline vs Simple Metrics

| algo | seeds | mean(loop_red) | mean(clipped) | mean(unique_cov) | mean(backtrack_rate) |
|------|------|----------------|---------------|------------------|--------------------|
| dfs | 2 | 1.000 | 1.000 | 1.000 | 0.000 |
| random | 2 | 16.362 | 16.362 | 0.144 | 0.468 |
| simple | 2 | 5.455 | 5.455 | 0.222 | 0.249 |

## Timing (Simple Aggregated)

| metric | mean_ms |
|--------|---------|
| timing_wiring_mean_ms | 120.345 |
| timing_snapshot_mean_ms | 3.577 |
| timing_gedig_mean_ms | 654.740 |
| timing_recall_mean_ms | 0.000 |

## Backtrack Plan Reasons (Simple)

| reason | count |
|--------|-------|