# Baseline vs Simple Metrics

| algo | seeds | mean(loop_red) | mean(clipped) | mean(unique_cov) | mean(backtrack_rate) |
|------|------|----------------|---------------|------------------|--------------------|
| dfs | 3 | 1.000 | 1.000 | 1.000 | 0.000 |
| random | 3 | 34.021 | 34.021 | 0.374 | 0.265 |
| simple | 3 | 2.766 | 2.766 | 0.742 | 0.108 |

## Timing (Simple Aggregated)

| metric | mean_ms |
|--------|---------|
| timing_wiring_mean_ms | 23.700 |
| timing_snapshot_mean_ms | 1.065 |
| timing_gedig_mean_ms | 131.352 |
| timing_recall_mean_ms | 0.000 |

## Backtrack Plan Reasons (Simple)

| reason | count |
|--------|-------|