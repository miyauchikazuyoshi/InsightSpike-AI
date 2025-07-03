# Experiment 2: 実際の動的データ成長実験

## 概要
InsightSpike-AI CLIを使用して、実際にデータを動的に追加し、成長を測定する実験

## 目的
- InsightSpike-AIシステムに実際にデータを追加
- graph_pyg.ptとepisodes.jsonの動的成長を確認
- 実際の圧縮率とパフォーマンスを測定

## ディレクトリ構造
```
experiment_2/
├── dynamic_growth/    # 動的成長の追跡データ
├── data_backup/      # 実験前のデータバックアップ
├── results/          # 実験結果
└── scripts/          # 実験スクリプト
```

## 実験手順
1. 現在のデータをバックアップ
2. HuggingFaceデータセットを準備
3. InsightSpike CLIを使用してデータを段階的に追加
4. 各段階でのファイルサイズと内容を記録
5. 成長パターンと圧縮効率を分析