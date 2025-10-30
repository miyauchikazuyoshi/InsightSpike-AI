# Baseline vs Simple Metrics

| algo | seeds | mean(loop_red) | mean(clipped) | mean(unique_cov) | mean(backtrack_rate) |
|------|------|----------------|---------------|------------------|--------------------|
| dfs | 1 | 1.000 | 1.000 | 1.000 | 0.000 |
| random | 1 | 1.271 | 1.271 | 0.880 | 0.095 |
| simple | 1 | 4.674 | 4.674 | 0.403 | 0.420 |

## Timing (Simple Aggregated)

| metric | mean_ms |
|--------|---------|
| timing_wiring_mean_ms | 7.877 |
| timing_snapshot_mean_ms | 0.789 |
| timing_gedig_mean_ms | 152.074 |
| timing_recall_mean_ms | 0.066 |

## Backtrack Plan Reasons (Simple)

| reason | count |
|--------|-------|