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

## Ablation (planned)
Note: AG/DG only affect expansion; answer generation is unconditional.

| Variant | Intent | Status |
|---|---|---|
| Full geDIG (AG+DG+F) | Default thresholds and expansions | baseline |
| AG only | Force expansion (lower `--theta-ag`, raise `--max-expansions`) | TODO |
| DG only | Disable expansion (`--max-expansions 0`) | TODO |
| F without ΔSP | Align with geDIG core config (TBD) | TODO |

## Expected outputs
- `experiments/hotpotqa-benchmark/results/gedig_*.jsonl`
- `experiments/hotpotqa-benchmark/results/gedig_*_summary.json`
- `docs/paper/data/hotpotqa_sample_summary.json`

## Known limitations
- Mock runs are not comparable to paper numbers; use a real LLM for full eval.
- Full dev set runs are slow and may hit API limits.
