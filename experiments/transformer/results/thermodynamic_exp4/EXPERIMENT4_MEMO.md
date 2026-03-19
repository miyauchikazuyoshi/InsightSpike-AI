# Experiment 4: F-Regularized Training — Provisional Results

**Date**: 2026-03-19
**Status**: Provisional (single run, needs replication)

## Setup

- Model: DistilBERT (distilbert-base-uncased)
- Task: SST-2 sentiment classification
- Samples: 2000 train, 500 eval
- Epochs: 3
- Device: MPS (Apple Silicon)
- Beta (F regularization weight): 0.1

## Three Conditions

| Condition | Loss Function | Hypothesis |
|-----------|--------------|------------|
| Baseline | CE only | Control |
| Positive | CE + beta * F (minimize F) | "Reducing F improves learning" |
| Negative | CE - beta * F (maximize F) | "Increasing F improves learning" |

## Results

| Condition | Epoch 1 | Epoch 2 | Epoch 3 (Final) |
|-----------|---------|---------|-----------------|
| Baseline | 89.4% | 89.4% | **88.1%** |
| Positive | 89.4% | 88.5% | **87.2%** (-0.9pp) |
| Negative | 89.7% | 89.4% | **89.4%** (+1.4pp) |

**Conclusion**: `negative_better`

## Interpretation (Provisional)

### Key Finding
F maximization (Negative) outperforms both Baseline and F minimization (Positive).

### What This Means

```
F = edge_cost - lambda * query_relevance (geDIG F-evaluation)

F minimization = "make all edges AG" = flatten attention structure
  -> Accuracy DROPS: model loses structural diversity in representations

F maximization = "preserve DG edges" = maintain attention heterogeneity
  -> Accuracy MAINTAINED/IMPROVED: model keeps "what it doesn't know" explicit
```

### Connection to AGHT (Analytical Heterogeneous Graph Transformer)

This result is consistent with today's AGHT findings:
- AGHT preserves DG edges as structural information (not penalty)
- QKV attention separates "what the query needs" (Q) from "what the node provides" (K)
- DG edges = low attention = information gaps = valuable structural signal

### Theoretical Implication

```
CE Loss: optimizes "get the right answer" (compression efficiency)
F regularization: preserves "knowledge structure" (information flow topology)

CE + F_max: optimize answers WHILE maintaining structural diversity
           = "learn the answer but don't collapse your representation"
```

This suggests F captures an independent axis of learning quality that CE alone misses.

## Caveats

1. **Single run** — needs replication with different seeds
2. **Small dataset** (2000 samples) — may not generalize to full-scale
3. **Beta=0.1 only** — sensitivity to beta not explored
4. **DistilBERT only** — needs testing on larger models
5. **SST-2 only** — needs testing on reasoning-intensive tasks

## Next Steps

- [ ] Replicate with 3+ random seeds for statistical significance
- [ ] Beta sweep: [0.01, 0.05, 0.1, 0.2, 0.5]
- [ ] Test on reasoning tasks (NLI, multi-hop QA) where DG structure matters more
- [ ] Visualize attention pattern differences between conditions
- [ ] Scale to full dataset and larger models
