# geDIG Logical Operations Analysis

**Date**: 2025-07-03 01:14:44

**Random Seed**: 42


## Key Findings

1. **Best Logical Operation**: Linear (Recall@5=0.190)
2. **vs Sentence-BERT**: 95.0% of SBERT performance
3. **Best AND Configuration**: geDIG-AND-Low (Recall@5=0.060)

## Logical Operations Comparison

| Operation | Recall@1 | Recall@5 | Recall@10 | Avg Hits | Description |
|-----------|----------|----------|-----------|----------|-------------|
| Linear | 0.020 | 0.190 | 0.330 | 5.9 | Weighted sum with threshold boost |
| AND | 0.000 | 0.030 | 0.080 | 1.4 | Both IG and GED must pass thresholds |
| OR | 0.000 | 0.040 | 0.140 | 24.6 | Either IG or GED must pass thresholds |
| WeightedAND | 0.010 | 0.100 | 0.290 | 20.9 | Graduated scoring based on threshold passing |
| Multiplicative | 0.010 | 0.100 | 0.270 | 1.4 | Score multiplied by threshold activations |

## Threshold Analysis (AND Operation)

| Threshold | IG | GED | Recall@5 | Hit Rate |
|-----------|-----|-----|----------|----------|
| Low | 0.2 | -0.15 | 0.060 | 4.3% |
| Medium | 0.3 | -0.1 | 0.030 | 0.3% |
| High | 0.4 | -0.05 | 0.010 | 0.0% |

## Insights

- AND operation is too restrictive (0.3% hit rate)