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

## Threshold sweep (mock)
Mock LLM run (dev=500, tune_size=200).

| Tune (AG/DG pct) | theta_ag | theta_dg | ag_fire_rate | dg_fire_rate | F1 | SF-F1 | Avg edges |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10/10 | -0.0914 | -0.0914 | 0.912 | 0.088 | 0.0608 | 0.2786 | 13.28 |
| 30/30 | 0.0859 | 0.0859 | 0.700 | 0.300 | 0.0611 | 0.3049 | 11.58 |
| 50/30 | 0.2855 | 0.0859 | 0.460 | 0.300 | 0.0616 | 0.3242 | 9.53 |
| 70/50 | 0.2881 | 0.2855 | 0.274 | 0.540 | 0.0629 | 0.3405 | 8.01 |

## Real LLM (small)
OpenAI run (dev=200, tuned on all 200).

| EM | F1 | SF-F1 | ag_fire_rate | dg_fire_rate | theta_ag | theta_dg | Notes |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.4050 | 0.5541 | 0.3094 | 0.700 | 0.300 | 0.0856 | 0.0856 | `LLM_PROVIDER=openai`, model=`gpt-4o-mini` |

## Expected outputs
- `experiments/hotpotqa-benchmark/results/gedig_*.jsonl`
- `experiments/hotpotqa-benchmark/results/gedig_*_summary.json`
- `experiments/hotpotqa-benchmark/results/threshold_sweep/*`
- `experiments/hotpotqa-benchmark/results/real_llm/*`
- `docs/paper/data/hotpotqa_sample_summary.json`
- `docs/paper/data/hotpotqa_ablation_dev_mock.json`

## Known limitations
- Mock runs are not comparable to paper numbers; use a real LLM for full eval.
- Full dev set runs are slow and may hit API limits.
