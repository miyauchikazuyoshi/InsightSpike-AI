# HotpotQA Benchmark Experiment Plan

## Overview

geDIG (Graph Edit Distance + Information Gain) の有効性を標準ベンチマーク HotpotQA で検証する。

**目標**: geDIG が multi-hop QA において動的グラフ更新の優位性を示すことを実証する。

---

## Timeline

```
Phase 0: JSAI投稿 (2週間以内)
├── 迷路実験中心の論文
└── HotpotQA 500件の予備結果を「今後の展望」として記載

Phase 1-5: 本実験 (JSAI後 1-2ヶ月)
├── HotpotQAフルベンチマーク
├── ベースライン比較
└── arXiv / 国際会議投稿
```

---

## Phase 0: JSAI向け予備実験

### 目標
- HotpotQA 500件でgeDIGの動作確認
- 最低1つのベースライン（BM25+GPT）との比較
- 論文に「標準ベンチマークでも検証中」と書ける状態

### タスク
- [ ] HotpotQA distractor dev set ダウンロード
- [ ] 500件サンプリング（固定seed）
- [ ] geDIG用フォーマット変換
- [ ] BM25+GPT-4o-mini ベースライン実装
- [ ] 実験実行・結果収集
- [ ] JSAI論文に1表追加

### 成果物
- `data/hotpotqa_500_sample.jsonl`
- `results/phase0_preliminary/`
- JSAI論文 Table X

---

## Phase 1: データ準備

### データセット
| 名前 | 設定 | 件数 | URL |
|------|------|------|-----|
| HotpotQA | distractor (dev) | 7,405 | https://hotpotqa.github.io/ |

### タスク
- [ ] HotpotQA distractor dev set ダウンロード
- [ ] データ形式確認・ドキュメント化
- [ ] geDIG用JSONL形式に変換
- [ ] train/dev/test split 確認（公式に従う）
- [ ] サンプル100件で動作確認

### データ形式（変換後）
```json
{
  "id": "5a8b57f25542995d1e6f1371",
  "question": "Which magazine was started first Arthur's Magazine or First for Women?",
  "answer": "Arthur's Magazine",
  "supporting_facts": [["Arthur's Magazine", 0], ["First for Women", 0]],
  "context": [["title1", ["sent1", "sent2", ...]], ...],
  "type": "comparison",
  "level": "medium"
}
```

### 成果物
- `data/hotpotqa_distractor_dev.jsonl` (7,405件)
- `data/hotpotqa_sample_100.jsonl` (動作確認用)
- `data/README.md` (データ形式ドキュメント)

---

## Phase 2: ベースライン実装

### 比較手法

| 手法 | 説明 | 優先度 |
|------|------|--------|
| BM25 + GPT-4o-mini | 古典検索 + LLM | 必須 |
| Contriever + GPT-4o-mini | Dense retriever | 必須 |
| Static GraphRAG | グラフ構築済み、更新なし | 必須 |
| ColBERT v2 | 高精度retriever | あれば良い |
| Self-RAG | 自己反省型RAG | あれば良い |

### 実装方針

#### BM25 + GPT-4o-mini
```python
# 使用ライブラリ: rank_bm25, openai
1. 全context文書をBM25インデックス化
2. questionでtop-k検索（k=5）
3. GPT-4o-miniで回答生成
```

#### Contriever + GPT-4o-mini
```python
# 使用ライブラリ: transformers (facebook/contriever)
1. 全context文書をembedding
2. questionとのcosine類似度でtop-k検索
3. GPT-4o-miniで回答生成
```

#### Static GraphRAG
```python
# 既存コード流用: experiments/rag-dynamic-db-v3-lite/external/graphrag
1. 全context文書からグラフ構築（1回のみ）
2. グラフからsubgraph検索
3. GPT-4o-miniで回答生成
# 更新なし = geDIGとの差分を明確化
```

### タスク
- [ ] BM25ベースライン実装
- [ ] Contrieverベースライン実装
- [ ] Static GraphRAG設定
- [ ] 共通評価インターフェース設計
- [ ] サンプル100件で動作確認

### 成果物
- `baselines/bm25_gpt.py`
- `baselines/contriever_gpt.py`
- `baselines/static_graphrag.py`
- `baselines/base.py` (共通インターフェース)

---

## Phase 3: geDIG適応

### 目標
HotpotQA の multi-hop QA タスクに geDIG を適応させる。

### 設計方針

```
HotpotQA Question
       ↓
[AG: Attention Gate]
  - 0-hop評価: 現在のグラフで回答可能か？
  - g₀ > θ_AG → 探索必要
       ↓
[Document Retrieval]
  - context文書からtop-k取得
  - グラフに候補ノード追加
       ↓
[DG: Decision Gate]
  - Multi-hop評価: 短絡形成されたか？
  - g_min < θ_DG → 統合確定
       ↓
[Answer Generation]
  - GPT-4o-miniで回答生成
```

### パラメータ
```yaml
lambda: 1.0          # 情報温度
gamma: 1.0           # 最短路重み
theta_ag: 0.4        # AGしきい値
theta_dg: 0.0        # DGしきい値
max_hops: 3          # Multi-hop評価の最大ホップ数
top_k_retrieval: 5   # 検索上位k件
```

### タスク
- [ ] HotpotQA用document ingestion実装
- [ ] AG/DGパラメータチューニング（サンプル100件）
- [ ] グラフ構築・更新ロジック確認
- [ ] エラーハンドリング強化
- [ ] サンプル100件で精度確認

### 成果物
- `src/hotpotqa_adapter.py`
- `configs/gedig_hotpotqa.yaml`
- `results/phase3_tuning/`

---

## Phase 4: フル実験

### 実験設定
```
データ: HotpotQA distractor dev (7,405件)
手法: 4種類
├── BM25 + GPT-4o-mini
├── Contriever + GPT-4o-mini
├── Static GraphRAG
└── geDIG (proposed)

Seeds: 3回 (42, 123, 456)
合計: 7,405 × 4 × 3 = 88,860 evaluations
```

### 評価指標
```python
metrics = {
    "em": exact_match,           # 完全一致
    "f1": token_f1,              # トークンF1
    "precision": token_precision,
    "recall": token_recall,
    "supporting_facts_em": sf_em,      # 根拠文の完全一致
    "supporting_facts_f1": sf_f1,      # 根拠文のF1
    "latency_p50": latency_median,
    "latency_p95": latency_95th,
    "api_calls": num_llm_calls,
    "graph_edges": num_edges,          # geDIG独自
    "compression_ratio": compression,  # geDIG独自
}
```

### 実行計画
```bash
# Phase 4a: 各手法を並列実行
python scripts/run_baseline.py --method bm25 --seed 42
python scripts/run_baseline.py --method contriever --seed 42
python scripts/run_baseline.py --method static_graphrag --seed 42
python scripts/run_gedig.py --seed 42

# Phase 4b: 結果集計
python scripts/aggregate_results.py --output results/phase4_full/
```

### タスク
- [ ] 実行スクリプト作成
- [ ] 並列実行環境設定
- [ ] 中間結果保存機能
- [ ] 進捗モニタリング
- [ ] 全手法実行（各seed）
- [ ] 結果集計・統計検定

### 成果物
- `scripts/run_baseline.py`
- `scripts/run_gedig.py`
- `scripts/aggregate_results.py`
- `results/phase4_full/*.jsonl`
- `results/phase4_full/summary.csv`

---

## Phase 5: 分析・執筆

### 分析項目

#### 5.1 主要結果
- EM/F1比較表（Table 1）
- 統計的有意差検定（paired t-test）
- 信頼区間

#### 5.2 詳細分析
- Question type別（comparison, bridge）の精度
- Difficulty level別（easy, medium, hard）の精度
- Multi-hop必要度とgeDIG効果の相関

#### 5.3 効率分析
- Latency比較
- API呼び出し回数比較
- グラフサイズ推移

#### 5.4 Case Study
- geDIGが勝つケースの具体例
- geDIGが負けるケースの具体例
- 失敗原因分析

### 図表リスト
```
Table 1: Main Results (EM, F1, SF-EM, SF-F1)
Table 2: Results by Question Type
Table 3: Results by Difficulty Level
Table 4: Efficiency Comparison (Latency, API calls)

Figure 1: geDIG Architecture for HotpotQA
Figure 2: EM/F1 by Question Type (bar chart)
Figure 3: Graph Size Evolution (line chart)
Figure 4: Latency Distribution (box plot)
Figure 5: Case Study Examples
```

### タスク
- [ ] 結果分析スクリプト作成
- [ ] 図表生成
- [ ] 統計検定実施
- [ ] Case study選定
- [ ] 論文本文執筆
- [ ] arXiv投稿

### 成果物
- `figures/*.pdf`
- `results/phase5_analysis/`
- `paper/hotpotqa_benchmark.tex`

---

## リソース見積もり

### 計算コスト
```
GPT-4o-mini: $0.15/1M input, $0.60/1M output
推定トークン:
  - 入力: ~1000 tokens/query × 7,405 × 4 × 3 = ~89M tokens
  - 出力: ~100 tokens/query × 7,405 × 4 × 3 = ~8.9M tokens
推定コスト: ~$13 (input) + ~$5 (output) = ~$20
バッファ込み: ~$50-100
```

### 時間見積もり
```
Phase 0: 1週間（JSAI投稿前）
Phase 1: 2-3日
Phase 2: 1週間
Phase 3: 1週間
Phase 4: 2-3日（実行時間）+ 1週間（バッファ）
Phase 5: 2週間

合計: 1-2ヶ月（副業ペース）
```

---

## ディレクトリ構成

```
experiments/hotpotqa-benchmark/
├── PLAN.md                 # この計画書
├── README.md               # 実験概要
├── data/
│   ├── README.md           # データ形式ドキュメント
│   ├── hotpotqa_distractor_dev.jsonl
│   ├── hotpotqa_sample_100.jsonl
│   └── hotpotqa_500_sample.jsonl
├── configs/
│   ├── gedig_hotpotqa.yaml
│   ├── bm25_baseline.yaml
│   ├── contriever_baseline.yaml
│   └── static_graphrag.yaml
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # HotpotQAローダー
│   ├── hotpotqa_adapter.py # geDIG適応
│   ├── evaluator.py        # 評価指標計算
│   └── utils.py
├── baselines/
│   ├── __init__.py
│   ├── base.py             # 共通インターフェース
│   ├── bm25_gpt.py
│   ├── contriever_gpt.py
│   └── static_graphrag.py
├── scripts/
│   ├── download_data.py    # データダウンロード
│   ├── run_baseline.py     # ベースライン実行
│   ├── run_gedig.py        # geDIG実行
│   ├── aggregate_results.py
│   └── generate_figures.py
├── results/
│   ├── phase0_preliminary/
│   ├── phase3_tuning/
│   ├── phase4_full/
│   └── phase5_analysis/
└── figures/
    └── .gitkeep
```

---

## チェックリスト

### Phase 0 (JSAI)
- [ ] HotpotQA 500件サンプル作成
- [ ] BM25+GPT ベースライン実行
- [ ] geDIG実行
- [ ] 結果をJSAI論文に追加

### Phase 1-5 (本実験)
- [ ] Phase 1: データ準備完了
- [ ] Phase 2: ベースライン実装完了
- [ ] Phase 3: geDIG適応完了
- [ ] Phase 4: フル実験完了
- [ ] Phase 5: 論文執筆完了
- [ ] arXiv投稿

---

## リスク管理

| リスク | 確率 | 影響 | 対策 |
|--------|------|------|------|
| geDIGがベースラインに負ける | 中 | 高 | 「どの条件で有効か」に論点を絞る |
| 計算時間超過 | 低 | 中 | サブセットで先に結果を出す |
| API代高騰 | 低 | 中 | GPT-4o-mini + キャッシュ活用 |
| バグ発覚 | 中 | 高 | サンプル100件で徹底検証 |
| HotpotQA形式と合わない | 低 | 中 | 早期に動作確認 |

---

## 次のアクション

1. **今すぐ**: `scripts/download_data.py` を実行してHotpotQAをダウンロード
2. **今週**: Phase 0のタスクを完了（JSAI用予備結果）
3. **JSAI後**: Phase 1-5を順次実行

---

## 参考資料

- [HotpotQA公式](https://hotpotqa.github.io/)
- [HotpotQA論文](https://arxiv.org/abs/1809.09600)
- [Contriever](https://github.com/facebookresearch/contriever)
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- [geDIG論文 (arxiv v6)](../docs/paper/arxiv_v6_en/)
