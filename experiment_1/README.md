# Experiment 1: 動的RAG成長実験と比較評価

## 実験概要
InsightSpike-AIの動的データ成長機能とRAG性能を検証する総合実験

## 実験構成

### 1. 動的RAG Embedding実験 (`dynamic_rag_growth/`)
- **目的**: CLIとDataフォルダを使用したgraph/jsonファイルの動的成長の確認
- **データセット**: HuggingFaceから取得済みの1000件データ
- **評価項目**:
  - データ圧縮率
  - メモリ使用効率
  - 動的成長の性能

### 2. 比較実験 (`comparison_experiment/`)
- **目的**: 動的成長させたRAGと従来型RAGの性能比較
- **評価項目**:
  - 回答精度検証
  - RAG応答速度検証
  - データ圧縮率比較

## ディレクトリ構造
```
experiment_1/
├── dynamic_rag_growth/     # 動的成長実験
│   ├── code/              # 実験コード
│   ├── data/              # 実験データ
│   ├── results/           # 実験結果
│   └── docs/              # ドキュメント
└── comparison_experiment/  # 比較実験
    ├── code/              # 比較実験コード
    ├── data/              # テストデータ
    ├── results/           # 比較結果
    └── docs/              # 分析レポート
```

## 実験ステップ
1. CLIを使用した動的データ追加の検証
2. 1000件データでの大規模動的成長実験
3. ベースラインRAGシステムの構築
4. 性能比較実験の実施
5. 結果分析とレポート作成