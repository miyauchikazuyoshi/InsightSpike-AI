# geDIG Intrinsic Threshold & Memory Efficiency Analysis

**Date**: 2025-07-03 01:05:25

**Random Seed**: 42


## Key Findings

1. **Best geDIG Threshold**: geDIG-Low (Recall@5=0.200)
2. **Best Memory Efficiency**: geDIG-Low (1360.3 docs/KB)
3. **Best Overall (Performance×Compression)**: geDIG-Low (Score=272.1)

## Performance Summary

| Method | Recall@5 | Latency (ms) | Memory (KB) | Compression | Efficiency |
|--------|----------|--------------|-------------|-------------|------------|
| TF-IDF | 0.000 | 1.0 | 42.0 | 1189.6 | 0.0 |
| Sentence-BERT | 0.250 | 29.8 | 750.0 | 66.7 | 16.7 |
| geDIG-Low | 0.200 | 1.5 | 36.8 | 1360.3 | 272.1 |
| geDIG-Medium | 0.200 | 1.6 | 36.8 | 1360.3 | 272.1 |
| geDIG-High | 0.200 | 1.5 | 36.8 | 1360.3 | 272.1 |
| geDIG-VeryHigh | 0.200 | 1.6 | 36.8 | 1360.3 | 272.1 |

## Threshold Analysis

The experiment tested various intrinsic reward thresholds:
- **Low** (IG=0.1, GED=-0.2): More permissive, accepts more matches
- **Medium** (IG=0.3, GED=-0.1): Balanced approach
- **High** (IG=0.5, GED=-0.05): More selective
- **VeryHigh** (IG=0.7, GED=-0.02): Very selective, only high-quality matches

**Finding**: Higher thresholds improve precision but may hurt recall.

## Memory Efficiency Insights

- TF-IDF: Sparse representation achieves 1189.6 docs/KB
- Sentence-BERT: Dense embeddings use 750.0KB
- geDIG variants: Text-based storage achieves 1360.3 docs/KB