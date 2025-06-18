# デバッグスクリプト

## 概要
実験・システム・データの状態確認・問題診断用スクリプト集

## 含まれるスクリプト

### 🔬 `debug_experiment_state.py`
- **機能**: 実験状態の詳細診断
- **用途**: 実験環境・設定・データ状態の確認
- **実行**: `python debug_experiment_state.py`

### 📊 `debug_graph_metrics.py`
- **機能**: グラフメトリクス（GED/IG）のデバッグ
- **用途**: GED急落現象・洞察検出ロジックの診断
- **実行**: `python debug_graph_metrics.py`

### 🌐 `debug_real_graphs.py`
- **機能**: 実際のグラフデータの詳細解析
- **用途**: リアルデータでのグラフ構造・接続性の診断
- **実行**: `python debug_real_graphs.py`

## 使用方法

```bash
# 実験状態診断
cd /path/to/InsightSpike-AI
python scripts/debugging/debug_experiment_state.py

# グラフメトリクス診断
python scripts/debugging/debug_graph_metrics.py

# リアルグラフデータ解析
python scripts/debugging/debug_real_graphs.py
```

## デバッグ対象

- **実験環境**: Poetry環境・依存関係・設定ファイル
- **データ状態**: グラフデータ・エピソード・メモリ状態
- **メトリクス**: GED/IG計算・洞察検出ロジック
- **接続性**: グラフ構造・ノード関係・エッジ分析

## トラブルシューティング

よくある問題と解決策は各スクリプト内のコメントを参照してください。

---
*InsightSpike-AI Project - Debugging Scripts*
