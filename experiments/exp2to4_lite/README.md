# Experiments II–IV (Lite)

This folder provides a minimal, cloud‑safe setup to (re)run and summarize:
- Experiment II: Static RAG baselines vs geDIG
- Experiment III: Dynamic GRAG × geDIG with PSZ metrics
- Experiment IV: Insight‑vector alignment (answer embeddings vs readout vectors)

It wraps the existing `experiments/rag-dynamic-db-v3-lite` pipeline and adds a small
alignment analyzer. No heavy deps or network required by default.

## Quick Start

1) Run Experiments II–III (lite, random embeddings)

```
python -m experiments.exp2to4_lite.src.run_experiment \
  --config experiments/exp2to4_lite/configs/exp23_geDIG_vs_baselines.yaml
```

- Writes a pipeline result JSON to `experiments/exp2to4_lite/results/`
- Prints a compact summary and saves `results/exp23_summary_<timestamp>.{json,csv}`

2) Run Experiment IV (alignment) using the above results

```
python -m experiments.exp2to4_lite.src.alignment \
  --results <path/to/pipeline_result.json> \
  --dataset experiments/exp2to4_lite/data/sample_queries_small.jsonl
```

- Outputs `results/<stem>_alignment.json` containing:
  - mean cosine similarities (support/all/random)
  - Δs (support−random, support−all), positive ratio, exact sign‑test p‑values, N

## Notes
- Default embedder is a lightweight, random fallback (no downloads). You can
  try a local model via `--embedding-model sentence-transformers/all-MiniLM-L6-v2`
  if already cached locally.
- All outputs are written under `results/` to be cloud‑safe.
- This does not include heavy Graph Transformer or GNN baselines; it targets the
  position‑paper metrics with similar trends.

## Paper‑Scale Suite

Generate 500‑query dataset, split, calibrate gates on val, and run test:

```
# Generate 500 queries (8 base domains; synthetic labels optional)
python experiments/exp2to4_lite/scripts/generate_dataset.py \
  --num-queries 500 \
  --output experiments/exp2to4_lite/data/sample_queries_500.jsonl

# Split into train/val/test (60/20/20)
python experiments/exp2to4_lite/scripts/split_dataset.py \
  --input experiments/exp2to4_lite/data/sample_queries_500.jsonl \
  --out-train experiments/exp2to4_lite/data/train_500.jsonl \
  --out-val   experiments/exp2to4_lite/data/val_500.jsonl \
  --out-test  experiments/exp2to4_lite/data/test_500.jsonl

# Calibrate theta_ag/theta_dg on val to target AG≈0.08, DG≈0.04, then run test
python -m experiments.exp2to4_lite.src.run_suite \
  --config experiments/exp2to4_lite/configs/exp23_paper.yaml --calibrate

# Summarize and compute alignment
python -m experiments.exp2to4_lite.run_exp23 \
  --config experiments/exp2to4_lite/configs/exp23_paper.yaml

python -m experiments.exp2to4_lite.src.alignment \
  --results <path/to/exp23_paper_*.json> \
  --dataset experiments/exp2to4_lite/data/test_500.jsonl
```

Tip: If you have a local sentence‑transformers model cached, set
`embedding.model` in the YAML to improve absolute scores.
