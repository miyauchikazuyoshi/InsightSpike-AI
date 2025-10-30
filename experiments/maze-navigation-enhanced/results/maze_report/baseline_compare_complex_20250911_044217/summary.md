# Baseline vs Simple Metrics

| algo | seeds | mean(loop_red) | mean(clipped) | mean(unique_cov) | mean(backtrack_rate) |
|------|------|----------------|---------------|------------------|--------------------|
| dfs | 3 | 1.000 | 1.000 | 1.000 | 0.000 |
| random | 3 | 3.427 | 3.427 | 0.673 | 0.196 |
| simple | 3 | 4.493 | 4.493 | 0.742 | 0.097 |

## Timing (Simple Aggregated)

| metric | mean_ms |
|--------|---------|
| timing_wiring_mean_ms | 14.855 |
| timing_snapshot_mean_ms | 0.965 |
| timing_gedig_mean_ms | 241.085 |
| timing_recall_mean_ms | 0.000 |

## Backtrack Plan Reasons (Simple)

| reason | count |
|--------|-------|