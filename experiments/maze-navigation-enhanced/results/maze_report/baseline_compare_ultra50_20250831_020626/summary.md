# Baseline vs Simple Metrics

| algo | seeds | mean(loop_red) | mean(clipped) | mean(unique_cov) | mean(backtrack_rate) |
|------|------|----------------|---------------|------------------|--------------------|
| dfs | 3 | 1.000 | 1.000 | 1.000 | 0.000 |
| random | 3 | 59.881 | 59.881 | 0.530 | 0.180 |
| simple | 3 | 11.014 | 11.014 | 0.602 | 0.124 |

## Timing (Simple Aggregated)

| metric | mean_ms |
|--------|---------|
| timing_wiring_mean_ms | 3.175 |
| timing_snapshot_mean_ms | 3.973 |
| timing_gedig_mean_ms | 4.985 |
| timing_recall_mean_ms | 0.000 |

## Backtrack Plan Reasons (Simple)

| reason | count |
|--------|-------|