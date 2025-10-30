# Baseline vs Simple Metrics

| algo | seeds | mean(loop_red) | mean(clipped) | mean(unique_cov) | mean(backtrack_rate) |
|------|------|----------------|---------------|------------------|--------------------|
| dfs | 3 | 1.000 | 1.000 | 1.000 | 0.000 |
| random | 3 | 2.207 | 2.207 | 0.734 | 0.143 |
| simple | 3 | 8.603 | 8.603 | 0.855 | 0.063 |

## Timing (Simple Aggregated)

| metric | mean_ms |
|--------|---------|
| timing_wiring_mean_ms | 14.248 |
| timing_snapshot_mean_ms | 1.454 |
| timing_gedig_mean_ms | 60.569 |
| timing_recall_mean_ms | 0.000 |

## Backtrack Plan Reasons (Simple)

| reason | count |
|--------|-------|