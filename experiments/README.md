# InsightSpike-AI 実験フレームワーク

## 📊 概要

InsightSpike-AIの段階的実験評価のための包括的フレームワークです。4つのフェーズに分けて体系的に性能評価・学術的検証を実施します。

## 🗂️ フォルダ構成

```
experiments/
├── phase1_dynamic_memory/          # Phase 1: 動的記憶構築実験
│   └── memory_construction_experiment.py
├── phase2_rag_benchmark/           # Phase 2: RAG比較実験  
│   └── rag_benchmark_experiment.py
├── phase3_gedig_maze/             # Phase 3: GEDIG迷路実験
│   └── gedig_maze_experiment.py
├── phase4_integrated_evaluation/  # Phase 4: 統合評価
│   └── integrated_evaluation_experiment.py
└── shared/                        # 共通ユーティリティ
    ├── __init__.py
    ├── benchmark_datasets.py      # データセット管理
    ├── evaluation_metrics.py      # 性能評価指標
    ├── experiment_reporter.py     # レポート生成
    └── environment_setup.py       # 実験環境構築
```

## � データ安全性機能

### 自動バックアップ・ロールバック
各実験は以下の安全性機能を備えています：

1. **実験前自動バックアップ**: データフォルダの完全バックアップを実験開始前に作成
2. **分離実験環境**: 実験は専用ディレクトリで実行、メインデータに影響なし
3. **実験後自動ロールバック**: 実験完了後、データフォルダを実験前の状態に自動復元
4. **選択的データコピー**: 実験に必要な最小限のデータのみを実験環境にコピー

### バックアップ管理
```bash
# バックアップ一覧表示
python -c "from experiments.shared.data_manager import DataStateManager; dm = DataStateManager(); print(dm.list_backups())"

# 手動バックアップ作成
python -c "from experiments.shared.data_manager import DataStateManager; dm = DataStateManager(); dm.create_backup('manual_backup', 'Manual backup before testing')"

# 特定バックアップからの復元
python -c "from experiments.shared.data_manager import DataStateManager; dm = DataStateManager(); dm.restore_backup('backup_id_here')"
```

### データフロー
```
プロジェクト/data/          # メインデータ（実験で変更されない）
    ├── processed/
    ├── embedding/
    └── models/

プロジェクト/data_backups/   # 自動バックアップ
    ├── pre_phase1_20250621_143022.tar.gz
    ├── pre_phase2_20250621_150315.tar.gz
    └── data_state_history.json

プロジェクト/experiment_data/ # 実験専用データ
    ├── phase1_memory_construction/
    │   ├── input/
    │   ├── processed/  # メインから複製
    │   ├── outputs/    # 実験結果
    │   └── temp/
    └── phase2_rag_benchmark/
        └── ... (同様の構造)
```

## �🚀 実行方法

### CLI機能付き実行（推奨）

```bash
# Phase 1: 動的記憶構築実験（CLI機能付き）
python experiments/phase1_dynamic_memory/memory_construction_experiment.py --help
python experiments/phase1_dynamic_memory/memory_construction_experiment.py --debug --quick
python experiments/phase1_dynamic_memory/memory_construction_experiment.py --sizes 100 500 --export json

# Phase 2: RAG比較実験（CLI機能付き）  
python experiments/phase2_rag_benchmark/rag_benchmark_experiment.py --help
python experiments/phase2_rag_benchmark/rag_benchmark_experiment.py --benchmarks ms_marco --quick
python experiments/phase2_rag_benchmark/rag_benchmark_experiment.py --rag-systems langchain llamaindex

# Phase 3: GEDIG迷路実験（CLI機能付き）
python experiments/phase3_gedig_maze/gedig_maze_experiment.py --help
python experiments/phase3_gedig_maze/gedig_maze_experiment.py --maze-sizes 10 20 --algorithms astar gedig
python experiments/phase3_gedig_maze/gedig_maze_experiment.py --quick --plot

# Phase 4: 統合評価（CLI機能付き）
python experiments/phase4_integrated_evaluation/integrated_evaluation_experiment.py --help
python experiments/phase4_integrated_evaluation/integrated_evaluation_experiment.py --meta-analysis --paper-format
```

### 個別フェーズ実行（基本モード）

```bash
# Phase 1: 動的記憶構築実験（自動バックアップ・ロールバック付き）
cd phase1_dynamic_memory
python memory_construction_experiment.py

# Phase 2: RAG比較実験  
cd phase2_rag_benchmark
python rag_benchmark_experiment.py

# Phase 3: GEDIG迷路実験
cd phase3_gedig_maze
python gedig_maze_experiment.py

# Phase 4: 統合評価
cd phase4_integrated_evaluation
python integrated_evaluation_experiment.py
```

**重要**: 各実験は以下の流れで安全に実行されます：
1. 🔄 実験前：dataフォルダの自動バックアップ
2. 📁 実験中：experiment_data/内で分離実行
3. 💾 実験後：結果保存 + データフォルダ自動復元

### 🎛️ CLI機能一覧

各Phase実験で利用可能な主要オプション：

| オプション | 説明 | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|-----------|------|---------|---------|---------|---------|
| `--help` | ヘルプ表示 | ✅ | ✅ | ✅ | ✅ |
| `--debug` | デバッグモード | ✅ | ✅ | ✅ | ✅ |
| `--quick` | クイックテスト | ✅ | ✅ | ✅ | ✅ |
| `--no-backup` | 高速モード | ✅ | ✅ | ✅ | ✅ |
| `--export json/csv/excel` | 出力形式 | ✅ | ✅ | ✅ | ✅ |
| `--config file.json` | 設定ファイル | ✅ | ✅ | ✅ | ✅ |
| `--sizes` | 文書サイズ指定 | ✅ | ❌ | ❌ | ❌ |
| `--benchmarks` | ベンチマーク選択 | ❌ | ✅ | ❌ | ❌ |
| `--maze-sizes` | 迷路サイズ指定 | ❌ | ❌ | ✅ | ❌ |
| `--meta-analysis` | メタ分析実行 | ❌ | ❌ | ❌ | ✅ |

詳細なCLI使用方法は [`docs/guides/EXPERIMENT_CLI_GUIDE.md`](../docs/guides/EXPERIMENT_CLI_GUIDE.md) を参照してください。

### 全フェーズ連続実行

```bash
# 全実験の順次実行（各実験後にデータが自動復元される）
for phase in phase1_dynamic_memory phase2_rag_benchmark phase3_gedig_maze phase4_integrated_evaluation; do
    echo "Starting $phase..."
    cd $phase
    python *.py
    cd ..
    echo "$phase completed with data restoration"
done
```

## 📋 各フェーズの概要

### Phase 1: 動的記憶構築実験
- **目的**: InsightSpike-AIの動的記憶構築機能の効率性・正確性検証
- **比較対象**: 標準RAGシステム、LangChain、LlamaIndex
- **主要指標**: 構築時間、メモリ使用量、検索精度、知識保持性

### Phase 2: RAG比較実験  
- **目的**: 主要RAGシステムとの包括的性能比較
- **ベンチマーク**: MS MARCO、Natural Questions、HotpotQA
- **主要指標**: 応答速度、メモリ効率、回答品質、FactScore

### Phase 3: GEDIG迷路実験
- **目的**: 粘菌アナロジーによるGEDIG評価での最適化性能検証  
- **比較アルゴリズム**: A*、Dijkstra、遺伝的アルゴリズム、強化学習
- **主要指標**: 経路最適性、計算効率、収束速度、メモリ使用量

### Phase 4: 統合評価
- **目的**: 全フェーズ結果の統合分析・メタ分析・論文用データ生成
- **出力**: 最終研究レポート、論文用図表、将来研究提案

## 🛠️ 依存関係

### 基本要件
```bash
pip install numpy pandas scikit-learn matplotlib seaborn plotly
pip install datasets transformers torch
```

### オプション要件（各フェーズ）
```bash
# Phase 2 RAG実験用
pip install langchain llamaindex haystack-ai

# Phase 3 迷路可視化用  
pip install networkx pygame

# GPU監視用（オプション）
pip install pynvml
```

## 📊 出力ファイル

各実験は以下の形式で結果を出力します：

```
{phase_name}_outputs/
├── experiment_results.json       # 実験結果データ
├── reports/
│   └── experiment_report_*.md   # 実験レポート  
├── visualizations/
│   ├── performance_comparison_*.html
│   ├── improvement_radar_*.html
│   └── *.png/*.html             # 各種グラフ
└── logs/
    └── experiment_*.log         # 実行ログ
```

## 🔍 評価指標一覧

### 共通指標
- **性能**: 実行時間、メモリ使用量、スループット
- **正確性**: 精度、再現率、F1スコア
- **効率性**: 計算コスト、リソース利用率
- **スケーラビリティ**: データサイズ対応性

### フェーズ固有指標
- **Phase 1**: 知識密度、検索遅延、記憶保持率
- **Phase 2**: BLEU、ROUGE、FactScore、応答関連性  
- **Phase 3**: 経路最適性、探索効率、収束安定性
- **Phase 4**: 統合スコア、効果サイズ、統計的有意性

## 📈 結果解釈

### 改善率の判定基準
- **有意改善**: 5%以上の改善
- **実質改善**: 15%以上の改善  
- **画期的改善**: 30%以上の改善

### 統計的有意性
- **p < 0.05**: 統計的有意
- **p < 0.01**: 高度に有意
- **p < 0.001**: 極めて有意

## 🔧 カスタマイズ

### 新しい実験の追加
1. `shared/`フォルダのユーティリティを活用
2. 標準的な実験構造に従う
3. 評価指標とレポート生成を統一

### ベンチマークデータセットの追加
`shared/benchmark_datasets.py`の`BenchmarkLoader`クラスに新しいデータセット読み込み関数を追加

### 評価指標の追加  
`shared/evaluation_metrics.py`の`MetricsCalculator`クラスに新しい指標計算関数を追加

## 🐛 トラブルシューティング

### よくある問題

1. **メモリ不足**
   ```bash
   # メモリ使用量を削減
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ```

2. **GPU利用不可**
   ```bash
   # CPU実行モードに切り替え
   export CUDA_VISIBLE_DEVICES=""
   ```

3. **依存関係エラー**
   ```bash
   # 仮想環境での実行を推奨
   python -m venv experiment_env
   source experiment_env/bin/activate
   pip install -r requirements.txt
   ```

## 📝 ログとデバッグ

### ログレベル設定
```python
import logging
logging.basicConfig(level=logging.DEBUG)  # 詳細ログ
logging.basicConfig(level=logging.INFO)   # 標準ログ
```

### デバッグモード
各実験スクリプトには`--debug`オプションが用意されています：
```bash
python experiment.py --debug
```

## 🔮 将来の拡張

### 計画中の機能
- 分散実験実行対応
- リアルタイム実験監視ダッシュボード
- 自動ハイパーパラメータ最適化
- クラウド実行環境対応

### 貢献方法
1. GitHub Issues での改善提案
2. Pull Requestでの機能追加
3. 実験結果・ベンチマークの共有

---

**重要**: 各実験は独立して実行可能ですが、Phase 4（統合評価）は他のフェーズの結果を必要とします。初回実行時は Phase 1-3 を先に完了させてください。
