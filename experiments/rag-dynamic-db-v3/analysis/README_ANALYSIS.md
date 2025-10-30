# RAG Analysis – Data Exports for Paper Figures (PSZ Scatter, Latency, Baselines, Human κ)

This folder provides exporters to derive paper‑ready CSVs for the RAG section. Outputs land under `data/rag_eval` or `data/latency` so the paper scripts can consume them directly.

## Targets and CSV Schemas

- PSZ Scatter points
  - Output: `data/rag_eval/psz_points.csv`
  - Columns: `run_id, query_id, H, k, PER, acceptance, FMR, latency_ms`

- Latency table (RAG)
  - Output: `data/latency/rag_latency.csv`
  - Columns: `run_id, H, k, latency_ms`

- Baseline summary (side‑by‑side)
  - Output: `data/rag_eval/baseline_summary.csv`
  - Columns: `method, acceptance, FMR, latency_ms`
  - (Optionally provide replicate rows `method, seed, acceptance...` and we’ll aggregate)

- Human acceptance agreement（Cohen’s κ）
  - Output: `data/rag_eval/human_acceptance.csv`
  - Columns: `query_id, annotator_id, accept`

## Exporters

- `export_psz_points.py`: derive PSZ散布図の点群（上記カラム）
- `export_latency.py`: 実測の追加レイテンシを集計して `rag_latency.csv` に出力
- `export_baseline_summary.py`: ベースラインの平均（or集計）を `baseline_summary.csv` に出力
- `export_human_acceptance.py`: 人手評価のフォーマット統一（κ計算用）

By default, exporters scan `experiments/rag-dynamic-db-v3/insight_eval/results/**` or accept `--from_json / --from_csv` to take explicit inputs.

## What to log (minimum)

- PSZ点群: 1行=1問の (PER, acceptance, FMR, latency_ms, H, k)。
- Latency: 1回の評価時間（追加レイテンシ; ms）と (H,k)。
- Baseline: 各method（Static/GraphRAG/DyG‑RAG/.../geDIG‑RAG）の (acceptance, FMR, latency_ms)。
- Human acceptance: `query_id × annotator_id` の 0/1 判定（κ計算用）。

