# Valid DistilGPT2 RAT Experiments

このディレクトリには、モックやチートを使用しない有効な実験ファイルのみが含まれています。

## 主要な実験ファイル

### 1. DistilGPT2実験
- `distilgpt2_rat_experiment.py` - メインのDistilGPT2 RAT実験
- `simple_distilgpt2_experiment.py` - シンプルなDistilGPT2実装
- `direct_distilgpt2_test.py` - DistilGPT2の直接テスト
- `complete_distilgpt2_test.py` - 完全なDistilGPT2テスト（正しい英英辞典データ使用）

### 2. InsightSpike実験
- `proper_insightspike_rat.py` - 適切なInsightSpike実装（"No mocks, no cheating"）
- `insightspike_poc_experiment.py` - InsightSpike概念実証（答えを含まない）
- `insightspike_style_experiment.py` - InsightSpikeスタイル実験（チート回避）
- `cycle_based_rat_experiment.py` - サイクルベースのアプローチ

### 3. 比較実験
- `rat_with_rag_comparison.py` - RAGとの比較実験
- `improved_rag_experiment.py` - 改良型RAG実験
- `rag_with_definitions.py` - 定義を使用したRAG

### 4. ユーティリティ
- `rebuild_proper_database.py` - 正しいデータベースの再構築
- `visualize_results.py` - 結果の可視化
- `visualize_rag_comparison.py` - RAG比較の可視化

## 削除されたファイル（モック/チート）
- ~~`fast_insightspike_experiment.py`~~ - モックLLMで答えをハードコード
- ~~`proper_db_builder.py`~~ - 答えを含むデータベース構築
- ~~`graphrag_comparison.py`~~ - 答えの接続を事前に含むグラフ

## 実験実行のガイドライン
1. 英英辞典の定義を使用（答えを直接含まない）
2. 実際のLLMを使用（モックなし）
3. 公正な比較のためベースラインを含める
4. 結果を透明に報告する