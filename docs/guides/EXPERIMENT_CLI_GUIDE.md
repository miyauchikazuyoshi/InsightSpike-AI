# 実験CLI機能ガイド

## 📋 概要

全Phase実験（Phase 1-4）にCLI機能を実装し、柔軟で使いやすい実験実行環境を提供します。

## 🚀 基本使用方法

### Phase 1: 動的記憶構築実験

```bash
# 基本実行
python experiments/phase1_dynamic_memory/memory_construction_experiment.py

# デバッグモードで実行
python experiments/phase1_dynamic_memory/memory_construction_experiment.py --debug

# 特定の文書サイズのみテスト
python experiments/phase1_dynamic_memory/memory_construction_experiment.py --sizes 100 500

# クイックテスト（小規模）
python experiments/phase1_dynamic_memory/memory_construction_experiment.py --quick

# 高速モード（バックアップなし）
python experiments/phase1_dynamic_memory/memory_construction_experiment.py --no-backup

# JSON形式で結果出力
python experiments/phase1_dynamic_memory/memory_construction_experiment.py --export json

# 設定ファイル使用
python experiments/phase1_dynamic_memory/memory_construction_experiment.py --config phase1_config.json

# 複数回実行で平均値計算
python experiments/phase1_dynamic_memory/memory_construction_experiment.py --runs 3

# ベースラインのみテスト
python experiments/phase1_dynamic_memory/memory_construction_experiment.py --baseline-only

# InsightSpikeのみテスト
python experiments/phase1_dynamic_memory/memory_construction_experiment.py --insightspike-only
```

### Phase 2: RAG比較実験

```bash
# 基本実行
python experiments/phase2_rag_benchmark/rag_benchmark_experiment.py

# 特定のベンチマークのみ実行
python experiments/phase2_rag_benchmark/rag_benchmark_experiment.py --benchmarks ms_marco natural_questions

# 特定のRAGシステムと比較
python experiments/phase2_rag_benchmark/rag_benchmark_experiment.py --rag-systems langchain llamaindex

# サンプルサイズ指定
python experiments/phase2_rag_benchmark/rag_benchmark_experiment.py --sample-size 50

# クイックテスト
python experiments/phase2_rag_benchmark/rag_benchmark_experiment.py --quick

# Excel形式で結果出力
python experiments/phase2_rag_benchmark/rag_benchmark_experiment.py --export excel
```

### Phase 3: GEDIG迷路実験

```bash
# 基本実行
python experiments/phase3_gedig_maze/gedig_maze_experiment.py

# 特定の迷路サイズでテスト
python experiments/phase3_gedig_maze/gedig_maze_experiment.py --maze-sizes 10 20 50

# 特定のアルゴリズムと比較
python experiments/phase3_gedig_maze/gedig_maze_experiment.py --algorithms astar dijkstra gedig

# 迷路生成数指定
python experiments/phase3_gedig_maze/gedig_maze_experiment.py --maze-count 5

# 可視化付きで実行
python experiments/phase3_gedig_maze/gedig_maze_experiment.py --plot

# クイックテスト（小規模）
python experiments/phase3_gedig_maze/gedig_maze_experiment.py --quick
```

### Phase 4: 統合評価実験

```bash
# 基本実行（自動的に前フェーズ結果を統合）
python experiments/phase4_integrated_evaluation/integrated_evaluation_experiment.py

# 特定の結果ディレクトリを統合
python experiments/phase4_integrated_evaluation/integrated_evaluation_experiment.py --previous-results path/to/phase1_results path/to/phase2_results

# メタ分析実行
python experiments/phase4_integrated_evaluation/integrated_evaluation_experiment.py --meta-analysis

# 論文用フォーマットで出力
python experiments/phase4_integrated_evaluation/integrated_evaluation_experiment.py --paper-format

# クイック統合（簡略分析）
python experiments/phase4_integrated_evaluation/integrated_evaluation_experiment.py --quick
```

## 🔧 共通オプション

すべてのPhaseで利用可能な共通オプション：

| オプション | 説明 | 例 |
|-----------|------|-----|
| `--debug` | デバッグモード（詳細ログ） | `--debug` |
| `--output` | 出力ディレクトリ指定 | `--output ./my_results` |
| `--export` | 結果エクスポート形式 | `--export json` |
| `--no-backup` | データバックアップスキップ | `--no-backup` |
| `--quick` | クイックテスト | `--quick` |
| `--config` | 設定ファイル使用 | `--config config.json` |
| `--no-report` | レポート生成スキップ | `--no-report` |
| `--plot` | グラフ生成 | `--plot` |

## 📝 設定ファイル例

### Phase 1設定ファイル例 (phase1_config.json)

```json
{
  "debug": false,
  "document_sizes": [50, 100, 200, 500, 1000],
  "num_runs": 3,
  "export_format": "json",
  "generate_report": true,
  "generate_plots": true,
  "baseline_only": false,
  "insightspike_only": false,
  "selective_copy": ["processed", "embedding", "models"]
}
```

### Phase 2設定ファイル例 (phase2_config.json)

```json
{
  "debug": false,
  "benchmarks": ["ms_marco", "natural_questions", "hotpot_qa"],
  "rag_systems": ["langchain", "llamaindex", "haystack"],
  "sample_size": 200,
  "export_format": "excel",
  "generate_report": true,
  "generate_plots": true
}
```

### Phase 3設定ファイル例 (phase3_config.json)

```json
{
  "debug": false,
  "maze_sizes": [10, 20, 50, 100, 200],
  "algorithms": ["astar", "dijkstra", "genetic", "reinforcement", "gedig"],
  "maze_count": 15,
  "export_format": "csv",
  "generate_report": true,
  "generate_plots": true
}
```

### Phase 4設定ファイル例 (phase4_config.json)

```json
{
  "debug": false,
  "meta_analysis": true,
  "paper_format": true,
  "export_format": "json",
  "generate_report": true,
  "generate_plots": true,
  "previous_results": [
    "experiment_data/phase1_memory_construction",
    "experiment_data/phase2_rag_benchmark", 
    "experiment_data/phase3_gedig_maze"
  ]
}
```

## 🔄 フォールバック機能

CLI機能が利用できない場合でも、自動的にフォールバックモードで実行されます：

```bash
# CLI機能エラー時の表示例
⚠️  CLI機能エラー: インポートエラー
🔧 基本モードで実行します
🔬 Phase 1: 動的記憶構築実験
==================================================
📊 文書サイズ: [50, 100, 200, 500]
🛡️  データバックアップ: 有効
🐛 デバッグモード: 無効
```

## 🔗 scripts/experiments/統合

各実験は`scripts/experiments/`の既存CLI機能との統合を試行し、利用できない場合は標準モードにフォールバックします：

```bash
# 統合成功時
✅ scripts/experiments/ExperimentCLI統合済み
✅ scripts/experiments/ExperimentRunner統合済み
✅ scripts/experiments/統合モードで実行完了

# 統合失敗時
⚠️  scripts統合モードエラー: インポートエラー
🔧 標準モードで実行します
```

## 📊 実験モード

### 🛡️ 安全モード（推奨）
- データの自動バックアップ
- 実験用データ分離
- 実験後の自動ロールバック

### ⚡ 高速モード
- バックアップスキップ
- 直接実行
- 開発・テスト用

### 🚧 クイックモード
- 小規模データセット
- 限定的な比較
- 動作確認用

## 🎯 使用例シナリオ

### 1. 開発・デバッグ時

```bash
# 高速デバッグ実行
python phase1_experiment.py --debug --quick --no-backup
```

### 2. 本格実験実行

```bash
# 完全な安全モードで実行
python phase1_experiment.py --config production_config.json --export json
```

### 3. 特定条件テスト

```bash
# 特定パラメータでの比較
python phase2_experiment.py --benchmarks ms_marco --rag-systems langchain --sample-size 50
```

### 4. 論文用データ生成

```bash
# 全フェーズ統合・論文フォーマット
python phase4_experiment.py --meta-analysis --paper-format --export excel
```

---

**重要**: 各実験は独立して実行可能ですが、フォールバック機能により安定した動作を保証します。CLI機能が利用できない環境でも基本モードで実行されます。
