# Cross-Domain Analogy QA (Structural Similarity)

## Claim
- Structural similarity enables cross-domain transfer, improving F1 by ~+0.60 on the synthetic QA set.

## Setup
- No external dataset (bundled under `experiments/structural_similarity/cross_domain_qa/data`).
- Optional: `INSIGHTSPIKE_LITE_MODE=1` for minimal imports.

## Run
```bash
make reproduce-analogy
```

## Metrics
- F1 mean, exact match, analogy detection rate
- Breakdown by difficulty and structure type

## Ablation
AG/DG gating is not used in this synthetic QA; ablation focuses on SS on/off.

Latest run (comparison_results.json, summarized in `docs/paper/data/analogy_ablation.json`).

| Variant | F1 mean | Exact match | Analogy detection | Notes |
|---|---:|---:|---:|---|
| Structural similarity on | 0.6603 | 0.1667 | 1.00 | `ss_enabled=True` |
| Structural similarity off | 0.0617 | 0.0000 | 0.00 | `ss_enabled=False` |
| Analogy weight off | - | - | - | N/A |

## Expected outputs
- `experiments/structural_similarity/cross_domain_qa/results/comparison_results.json`
- `docs/paper/data/analogy_comparison.json`
- `docs/paper/data/analogy_ablation.json`

## Known limitations
- Synthetic dataset with template-based answer transfer.
- Threshold selection affects F1 and detection rates.
