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

## Ablation (planned)
AG/DG gating is not used in this synthetic QA; ablation focuses on SS on/off.

| Variant | Intent | Status |
|---|---|---|
| Structural similarity on | `ss_enabled=True` (default) | baseline |
| Structural similarity off | `ss_enabled=False` | TODO |
| Analogy weight off | `analogy_weight=0` (config) | TODO |

## Expected outputs
- `experiments/structural_similarity/cross_domain_qa/results/comparison_results.json`
- `docs/paper/data/analogy_comparison.json`

## Known limitations
- Synthetic dataset with template-based answer transfer.
- Threshold selection affects F1 and detection rates.
