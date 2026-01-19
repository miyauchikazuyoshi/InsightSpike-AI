# HotpotQA Benchmark

## Claim
- geDIG dynamic RAG improves EM on HotpotQA (reported +3.5pt).

## Setup
- Dataset: `experiments/hotpotqa-benchmark/data/`
- Download: `python experiments/hotpotqa-benchmark/scripts/download_data.py`
- For real runs, set `OPENAI_API_KEY` and `LLM_PROVIDER=openai`.
- For no-network runs, keep `LLM_PROVIDER=mock` (default in reproduce target).

## Run
```bash
make reproduce-hotpotqa
```

## Metrics
- EM/F1, SF-F1, latency
- Gate fire rates (initial/final)

## Ablation
Note: AG/DG only affect expansion; answer generation is unconditional.

Mock LLM run (dev=7,405, no-network).

| Variant | EM | F1 | SF-F1 | ag_fire_rate | dg_fire_rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| Full geDIG (AG+DG+F) | 0.0005 | 0.0610 | 0.3497 | 0.00 | 0.1236 | default thresholds |
| AG only | 0.0004 | 0.0604 | 0.2106 | 1.00 | 0.0000 | `--theta-ag -1 --theta-dg -10 --max-expansions 2` |
| DG only | 0.0005 | 0.0610 | 0.3497 | 0.00 | 0.1236 | `--max-expansions 0` |
| F without ΔSP | - | - | - | - | - | N/A (SP term not used in this adapter) |

## Expected outputs
- `experiments/hotpotqa-benchmark/results/gedig_*.jsonl`
- `experiments/hotpotqa-benchmark/results/gedig_*_summary.json`
- `docs/paper/data/hotpotqa_sample_summary.json`
- `docs/paper/data/hotpotqa_ablation_dev_mock.json`

## Known limitations
- Mock runs are not comparable to paper numbers; use a real LLM for full eval.
- Full dev set runs are slow and may hit API limits.
