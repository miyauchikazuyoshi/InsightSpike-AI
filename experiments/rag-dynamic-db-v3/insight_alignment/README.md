# Insight Alignment Pipeline (Supplemental Experiment)

This subdirectory packages the scripts needed to reproduce「洞察ベクトルと LLM 応答の一致度判定」補足実験。既存の `insight_eval/run_alignment.py`（論文付属スクリプト）を呼び出しやすいように、RAG 実験結果から必要なフォーマットを生成するユーティリティを追加しています。

## フォルダ構成

```
insight_alignment/
  README.md
  __init__.py
  prepare_alignment_inputs.py   # RAG の QA ログを alignment 用 JSON へ変換
  run_alignment_pipeline.py     # 上記 + docs/paper/run_insight_vector_alignment.py を一括実行
  inputs/                       # 生成される alignment 入力ファイルの既定保存先
  outputs/                      # Δs 指標や図表などの出力先
```

## 事前準備

1. RAG 実験のログから `{"question": "...", "answer": "..."}` 形式の JSONL を作成します。  
   例: `rag_results/run123_qa.jsonl`
2. Sentence-BERT を利用する場合は `pip install sentence-transformers` を済ませてください（論文本体の要件 A1–A3 に沿った埋め込み空間を確保するため）。
3. 既定の洞察知識ベースは `data/insight_store/knowledge_base/initial/insight_dataset.txt` を参照します。独自 KB を使用する場合は `--kb` オプションでパスを指定できます。

## 使い方

```bash
cd experiments/rag-dynamic-db-v3/insight_alignment

# ステップ1: QA ログから alignment 用 JSON を生成
python prepare_alignment_inputs.py \
  --qa-jsonl ../some_run/qa_pairs.jsonl \
  --output inputs/alignment_inputs.json

# ステップ2: Δs 算出と図表出力（docs/paper/run_insight_vector_alignment.py を呼び出します）
python run_alignment_pipeline.py \
  --qa-jsonl ../some_run/qa_pairs.jsonl \
  --outdir outputs/run123 \
  --H 3 --gamma 0.7 --tau 0.35
```

`run_alignment_pipeline.py` は内部で `prepare_alignment_inputs.py` を呼び、続いて `insight_eval/run_alignment.py`（→ `docs/paper/run_insight_vector_alignment.py`）を実行します。出力として以下が生成されます。

- `outputs/run123/alignment_inputs.json` : alignment スクリプト用 JSON
- `outputs/run123/rag_insight_alignment.pdf` : Δs 分布ヒストグラム
- `outputs/run123/rag_insight_alignment_stats.json` : Δs、負例比較、符号検定などの統計

## オプション

`run_alignment_pipeline.py` は `--kb`, `--H`, `--gamma`, `--tau`, `--residual-ans`, `--neg` 等、論文スクリプトと同じ引数を透過的に受け付けます。詳細は `python run_alignment_pipeline.py --help` を参照してください。
