# geDIG vs RAG/graphRAG/MA-RAG — Comparison Harness (MVP)

This directory provides a minimal, reproducible harness to compare four retrieval-generation pipelines on the same corpus and queries under consistent constraints:

- Plain RAG (static; no updates)
- Heuristic graphRAG (cosine-based, frequency-based)
- geDIG‑graphRAG (proposed; update gating via F=ΔGED−λΔIG)
- (Optional) MA‑RAG placeholder (future; 1-iteration Planner→Retriever→Critic)

The MVP focuses on unified logging and export (CSV/JSON) so figures and tables can be generated for the paper.

## What it does

- Loads a small, multi‑domain knowledge base + query list from `experiments/rag-dynamic-db-v3/src/multidomain_knowledge_base.py`
- Builds four systems using existing baselines in `src/baselines/`
- Runs the same queries through each system, with the same embedder/generator placeholders
- Logs per‑query timings, update decisions, graph sizes
- Exports per‑method summaries (update rate, avg latency, graph stats)

Note: This MVP uses repository placeholders (DummyEmbedder/DummyGenerator) for speed and determinism. You can swap in a real embedder/LLM later without changing the experiment logic.

## Quick start

```
cd experiments/rag-dynamic-db-v3
PYTHONPATH=src python compare/bench_compare.py \
  --outdir experiments/rag-dynamic-db-v3/results/compare_mvp \
  --n-queries 30 --seed 42
```

Artifacts are written to `--outdir`:

- `runs/run_YYYYmmdd_HHMMSS/`
  - `summary.csv` — per‑method aggregate stats
  - `records.csv` — per‑query records (method, timings, decisions, graph sizes)
  - `graphs/{method}.graph.json` — final graph snapshots
  - `state/{method}.state.json` — method state with response history

## Notes

- “graphRAG”の代表として、簡易ヒューリスティックの2種（cosine/frequency）を並置。どちらも KG 更新とサブグラフ接続を行うため、静的RAGとの対比が可能です。
- geDIG は `baselines/gedig_rag.py` を使用。受容/FMR/PSZ といった運用指標に必要な CSV は、既存の insight_eval パイプと組み合わせてください。
- MA‑RAG は将来の差し込み位置を空けています（1反復固定の最小構成）。

## Extend

- Replace DummyEmbedder / DummyGenerator with real components from `insightspike` or external providers.
- Add quality metrics (EM/F1/Faithfulness/Attribution) when gold answers and evidence IDs are available.
- Add ablations: ΔIGのみ/ΔGEDのみ/ランダム/中央性ヒューリスティック等。

