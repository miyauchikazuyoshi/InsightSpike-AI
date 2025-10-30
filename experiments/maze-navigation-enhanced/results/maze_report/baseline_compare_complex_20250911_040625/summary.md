# Baseline vs Simple Metrics

| algo | seeds | mean(loop_red) | mean(clipped) | mean(unique_cov) | mean(backtrack_rate) |
|------|------|----------------|---------------|------------------|--------------------|
| dfs | 3 | 1.000 | 1.000 | 1.000 | 0.000 |
| random | 3 | 4.229 | 4.229 | 0.674 | 0.172 |
| simple | 3 | 3.136 | 3.136 | 0.723 | 0.114 |

## Timing (Simple Aggregated)

| metric | mean_ms |
|--------|---------|
| timing_wiring_mean_ms | 16.554 |
| timing_snapshot_mean_ms | 1.018 |
| timing_gedig_mean_ms | 271.254 |
| timing_recall_mean_ms | 0.000 |

## Backtrack Plan Reasons (Simple)

| reason | count |
|--------|-------|