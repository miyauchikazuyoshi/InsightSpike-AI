# HotpotQA Benchmark for geDIG

geDIG (Graph Edit Distance + Information Gain) の標準ベンチマーク評価実験。

## Quick Start

```bash
# 1. データダウンロード
python scripts/download_data.py

# 2. サンプル実行（動作確認）
python scripts/run_gedig.py --data data/hotpotqa_sample_100.jsonl

# 3. ベースライン実行
python scripts/run_baseline.py --method bm25 --data data/hotpotqa_sample_100.jsonl
# closed-book（質問のみ）ベースライン
python scripts/run_baseline.py --method closed_book --data data/hotpotqa_sample_100.jsonl

# 4. フル実験
python scripts/run_gedig.py --data data/hotpotqa_distractor_dev.jsonl
```

## Result Aggregation & Figures

```bash
python scripts/aggregate_results.py --results-dir results
python scripts/generate_figures.py --summary results/summary_by_method.json
```

## Latest Comparison (by dataset)

```bash
python scripts/generate_latest_comparison.py
python scripts/generate_latest_figures.py --input results/latest_comparison.csv
```

Note: closed-book runs over the full dev set may hit OpenAI daily request limits.
If that happens, rerun after the limit resets (the partial JSONL can be reused for analysis).

## Threshold Tuning (Optional)

```bash
python scripts/run_gedig.py --data data/hotpotqa_sample_100.jsonl --tune-thresholds
```

## geDIG Gate Metrics

- `ag_fire_rate` / `dg_fire_rate` use the initial gate decisions (pre-expansion).
- `final_ag_fire_rate` / `final_dg_fire_rate` capture the final gates after expansion.
- TF-IDF hashed features are enabled via `gedig.tfidf_dim` (set to 0 to disable).

## Directory Structure

```
hotpotqa-benchmark/
├── PLAN.md           # 詳細計画書
├── README.md         # このファイル
├── data/             # データセット
├── configs/          # 設定ファイル
├── src/              # ソースコード
├── baselines/        # ベースライン実装
├── scripts/          # 実行スクリプト
├── results/          # 実験結果
└── figures/          # 図表
```

## Phases

| Phase | 内容 | 状態 |
|-------|------|------|
| 0 | JSAI向け予備実験 (500件) | TODO |
| 1 | データ準備 | TODO |
| 2 | ベースライン実装 | TODO |
| 3 | geDIG適応 | TODO |
| 4 | フル実験 | TODO |
| 5 | 分析・執筆 | TODO |

## Requirements

```bash
pip install rank-bm25 openai transformers datasets pyyaml
# Optional (Contriever/GPU + figure generation)
pip install torch matplotlib
```

## Closed-World Evaluation

本実験は HotpotQA の各例に付属する context のみを検索対象とする「閉世界」設定です。
BM25/Contriever/Static GraphRAG/geDIG はすべて各例の context 内で検索します。

## Contact

miyauchikazuyoshi@gmail.com
