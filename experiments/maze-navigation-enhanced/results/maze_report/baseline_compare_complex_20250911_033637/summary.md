# Baseline vs Simple Metrics

| algo | seeds | mean(loop_red) | mean(clipped) | mean(unique_cov) | mean(backtrack_rate) |
|------|------|----------------|---------------|------------------|--------------------|
| dfs | 1 | 1.000 | 1.000 | 1.000 | 0.000 |
| random | 1 | 1.271 | 1.271 | 0.880 | 0.095 |
| simple | 1 | 2.785 | 2.785 | 0.851 | 0.067 |

## Timing (Simple Aggregated)

| metric | mean_ms |
|--------|---------|
| timing_wiring_mean_ms | 22.170 |
| timing_snapshot_mean_ms | 2.073 |
| timing_gedig_mean_ms | 350.618 |
| timing_recall_mean_ms | 0.000 |

## Backtrack Plan Reasons (Simple)

| reason | count |
|--------|-------|