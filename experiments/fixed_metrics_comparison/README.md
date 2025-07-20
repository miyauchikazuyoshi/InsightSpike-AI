# Fixed Metrics Comparison Experiment

## 実験概要
This experiment validates the improved GED (Graph Edit Distance) and IG (Information Gain) implementation in InsightSpike after fixing the metrics calculation issues.

## 目的
1. Verify that the new GED/IG implementation produces meaningful similarity scores
2. Compare InsightSpike performance with baseline methods using corrected metrics
3. Demonstrate that the fixed metrics lead to better insight discovery

## 仮説
The corrected GED/IG implementation will:
- Produce similarity scores in a reasonable range (not near 0)
- Better capture semantic relationships between concepts
- Lead to more accurate insight discovery compared to baseline methods

## 方法
1. **Baseline Methods**:
   - Simple keyword matching
   - Basic cosine similarity
   - Standard RAG approach

2. **InsightSpike with Fixed Metrics**:
   - Using improved GED calculation with proper normalization
   - Using corrected IG calculation with appropriate entropy measures
   - Dynamic graph optimization based on meaningful similarity scores

3. **Test Scenarios**:
   - Cross-domain knowledge connections (e.g., quantum-biology)
   - Abstract concept relationships
   - Multi-hop reasoning tasks

## 評価指標
- Similarity score distribution (should be well-distributed, not clustered near 0)
- Insight quality (human evaluation on 1-5 scale)
- Response relevance and coherence
- Computational efficiency

## データセット
- Sample texts from various domains (quantum physics, biology, consciousness)
- Curated question-answer pairs testing cross-domain understanding
- Knowledge graph with known relationships for validation

## 成功基準
1. GED similarity scores show meaningful distribution (not all near 0)
2. InsightSpike outperforms baselines in insight quality by at least 20%
3. Discovered insights demonstrate valid cross-domain connections

## 実験実行手順
```bash
# 1. Setup experiment data
poetry run python src/setup_experiment.py

# 2. Run baseline comparison
poetry run python src/run_baseline_comparison.py

# 3. Run InsightSpike with fixed metrics
poetry run python src/run_fixed_metrics_experiment.py

# 4. Generate visualizations and report
poetry run python src/generate_report.py
```

## ディレクトリ構造
```
fixed_metrics_comparison/
├── src/                          # 実験プログラム
│   ├── setup_experiment.py       # データ準備
│   ├── run_baseline_comparison.py # ベースライン実行
│   ├── run_fixed_metrics_experiment.py # 改善版実行
│   ├── metrics_validator.py      # メトリクス検証
│   └── generate_report.py        # レポート生成
├── data/
│   ├── input/                    # 入力データ（読み取り専用）
│   │   ├── sample_texts.json
│   │   ├── test_questions.json
│   │   └── knowledge_graph.json
│   └── processed/                # 処理済みデータ
├── results/
│   ├── metrics/                  # 評価指標
│   ├── outputs/                  # 出力ファイル
│   └── visualizations/           # グラフなど
├── data_snapshots/               # データバックアップ
└── README.md                     # この文書
```

## 関連ドキュメント
- `/docs/development/GED_IG_REFACTORING_COMPLETE.md` - メトリクス修正の詳細
- `/src/insightspike/algorithms/graph_edit_distance.py` - 改善されたGED実装
- `/src/insightspike/algorithms/information_gain.py` - 改善されたIG実装

## 実験ログ
- 2025-07-20: 実験環境セットアップ開始