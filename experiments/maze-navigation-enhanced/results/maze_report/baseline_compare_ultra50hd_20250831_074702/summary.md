# Baseline vs Simple Metrics

| algo | seeds | mean(loop_red) | mean(clipped) | mean(unique_cov) | mean(backtrack_rate) |
|------|------|----------------|---------------|------------------|--------------------|
| dfs | 1 | 1.000 | 1.000 | 1.000 | 0.000 |
| random | 1 | 17.347 | 17.347 | 0.089 | 0.480 |
| simple | 1 | 9.828 | 9.828 | 0.165 | 0.205 |

## Timing (Simple Aggregated)

| metric | mean_ms |
|--------|---------|
| timing_wiring_mean_ms | 2.392 |
| timing_snapshot_mean_ms | 6.295 |
| timing_gedig_mean_ms | 7.232 |
| timing_recall_mean_ms | 0.167 |

## Backtrack Plan Reasons (Simple)

| reason | count |
|--------|-------|
| global_recall | 12 |