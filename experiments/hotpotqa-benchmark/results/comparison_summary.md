# HotPotQA Benchmark Results Summary

**Last Updated**: 2026-01-16

## Main Results

| Method | Dataset Size | EM | F1 | SF-F1 | P50 (ms) | Notes |
|--------|--------------|-----|------|-------|----------|-------|
| Closed-book | 500 | 0.222 | 0.353 | 0.000 | 656 | No retrieval |
| Contriever | 499 | 0.253 | 0.403 | 0.163 | 6,866 | Dense retriever |
| **BM25** | **7,405** | **0.366** | **0.523** | **0.350** | **820** | Sparse retriever |
| **geDIG** | **7,405** | **0.375** | **0.538** | **0.304** | **873** | Proposed |

## Key Findings

### 1. geDIG vs BM25 (Full Dataset Comparison)

| Metric | BM25 | geDIG | Diff | % Change |
|--------|------|-------|------|----------|
| EM | 0.366 | 0.375 | +0.009 | **+2.4%** |
| F1 | 0.523 | 0.538 | +0.015 | **+2.9%** |
| Recall | 0.611 | 0.635 | +0.024 | **+3.9%** |
| SF-F1 | 0.350 | 0.304 | -0.046 | -13.1% |
| P50 (ms) | 820 | 873 | +53 | +6.5% |

### 2. Observations

**Positive**:
- geDIG achieves higher EM and F1 than BM25
- Recall improvement (+3.9%) suggests better coverage
- Latency overhead is acceptable (+53ms, +6.5%)

**Concerning**:
- Supporting Facts F1 is lower (-13.1%)
  - geDIG may be retrieving correct answers but not the right evidence
  - Needs investigation

**Incomplete**:
- Contriever only ran on 500 samples (needs full dataset run)

## geDIG Gate Statistics

```
AG Fire Rate (initial): 54.6%
AG Fire Rate (final):   12.9%
DG Fire Rate (initial): 26.9%
DG Fire Rate (final):   81.5%

Avg geDIG Score: 0.780
Avg Graph Edges: 10.35
```

**Interpretation**:
- AG fires frequently initially but many are resolved without expansion
- DG confirms ~81% of candidates after evaluation
- Graph structure is being actively managed

## Next Steps

1. [ ] Run Contriever on full dataset (7,405)
2. [ ] Investigate SF-F1 drop - why is evidence retrieval worse?
3. [ ] Ablation: AG-only vs DG-only vs Full geDIG
4. [ ] Error analysis: Where does geDIG fail?
5. [ ] Parameter tuning: Can we improve SF-F1 without hurting EM/F1?

## Configuration Details

### BM25
```yaml
retrieval:
  method: bm25
  top_k: 5
llm:
  model: gpt-4o-mini
  temperature: 0.0
```

### geDIG
```yaml
lambda_weight: 0.5
theta_ag: 0.830
theta_dg: 0.828
top_k: 5
max_expansions: 1
max_hops: 3
gamma: 1.0
tfidf_dim: 64
llm:
  model: gpt-4o-mini
  temperature: 0.0
```

## Raw Data Files

- BM25: `bm25_20260115_210922_summary.json`
- geDIG: `gedig_20260115_115335_summary.json`
- Contriever: `contriever_20260113_161057_summary.json`
- Closed-book: `closed_book_20260113_211718_summary.json`
