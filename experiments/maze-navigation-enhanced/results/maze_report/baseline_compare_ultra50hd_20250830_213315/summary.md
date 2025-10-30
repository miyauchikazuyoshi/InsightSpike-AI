# Baseline vs Simple Metrics

| algo | seeds | mean(loop_red) | mean(clipped) | mean(unique_cov) | mean(backtrack_rate) |
|------|------|----------------|---------------|------------------|--------------------|
| dfs | 1 | 1.000 | 1.000 | 1.000 | 0.000 |
| random | 1 | 1.308 | 1.308 | 0.765 | 0.160 |
| simple | 1 | 1.000 | 1.000 | 1.000 | 0.000 |

## Timing (Simple Aggregated)

| metric | mean_ms |
|--------|---------|
| timing_wiring_mean_ms | 2.353 |
| timing_snapshot_mean_ms | 0.370 |
| timing_gedig_mean_ms | 28.496 |
| timing_recall_mean_ms | 0.001 |

## Backtrack Plan Reasons (Simple)

| reason | count |
|--------|-------|