# Baseline vs Simple Metrics

| algo | seeds | mean(loop_red) | mean(clipped) | mean(unique_cov) | mean(backtrack_rate) |
|------|------|----------------|---------------|------------------|--------------------|
| dfs | 20 | 1.000 | 1.000 | 1.000 | 0.000 |
| random | 20 | 8.316 | 8.316 | 0.600 | 0.188 |
| simple | 20 | 10.095 | 10.095 | 0.691 | 0.117 |

## Timing (Simple Aggregated)

| metric | mean_ms |
|--------|---------|
| timing_wiring_mean_ms | 34.218 |
| timing_snapshot_mean_ms | 2.140 |
| timing_gedig_mean_ms | 511.789 |
| timing_recall_mean_ms | 0.000 |

## Backtrack Plan Reasons (Simple)

| reason | count |
|--------|-------|