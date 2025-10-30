# Baseline vs Simple Metrics

| algo | seeds | mean(loop_red) | mean(clipped) | mean(unique_cov) | mean(backtrack_rate) |
|------|------|----------------|---------------|------------------|--------------------|
| dfs | 1 | 1.000 | 1.000 | 1.000 | 0.000 |
| random | 1 | 1.271 | 1.271 | 0.880 | 0.095 |
| simple | 1 | 4.391 | 4.391 | 0.802 | 0.070 |

## Timing (Simple Aggregated)

| metric | mean_ms |
|--------|---------|
| timing_wiring_mean_ms | 6.984 |
| timing_snapshot_mean_ms | 1.956 |
| timing_gedig_mean_ms | 135.120 |
| timing_recall_mean_ms | 0.000 |

## Backtrack Plan Reasons (Simple)

| reason | count |
|--------|-------|