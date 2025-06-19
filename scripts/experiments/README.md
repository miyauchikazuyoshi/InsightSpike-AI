# 実験管理システム利用ガイド

InsightSpike-AIの新しい実験管理システムは、同一条件での対照実験を可能にし、reproducibleで比較可能な結果を提供します。

## 🚀 クイックスタート

### 1. 高速比較実験（推奨）
最初のテストには、高速比較実験を実行してください：

```bash
# 統合コマンドを使用
python scripts/experiments/exp.py workflow quick_comparison

# または直接実行
python scripts/experiments/automated_workflow.py quick_comparison
```

この実験では3つの異なる設定で100エピソードずつ実行し、自動的に比較レポートを生成します。

### 2. パラメータ感度分析
パラメータの影響を詳細に調べたい場合：

```bash
python scripts/experiments/exp.py workflow parameter_sensitivity
```

### 3. 初期メモリサイズの影響調査
初期メモリサイズが洞察検出に与える影響を調査：

```bash
python scripts/experiments/exp.py workflow memory_initialization_study
```

## 📋 基本的な実験管理コマンド

### データ状態確認
```bash
# 現在のデータ状態を確認
python scripts/experiments/exp.py status

# データ整合性チェック
python scripts/experiments/exp.py check
```

### データ管理
```bash
# データクリーンアップ
python scripts/experiments/exp.py clean

# データバックアップ
python scripts/experiments/exp.py backup before_experiment

# バックアップ一覧
python scripts/experiments/exp.py list-backups

# データ復元
python scripts/experiments/exp.py restore before_experiment_20250618_230000
```

### 初期メモリ構築
```bash
# 50エピソードの初期メモリを構築
python scripts/experiments/exp.py build-memory --episodes 50 --seed 42

# 初期メモリなし（0エピソード）
python scripts/experiments/exp.py build-memory --episodes 0
```

## 🔬 カスタム実験の実行

### 1. 実験セッション作成
```bash
python scripts/experiments/exp.py create-session my_experiment_session
```

### 2. 初期メモリ構築
```bash
python scripts/experiments/exp.py build-memory --episodes 50 --seed 42
```

### 3. 単発実験実行
```bash
python scripts/experiments/exp.py run my_experiment_session baseline_test \
  --episodes 300 \
  --seed 42 \
  --ged-threshold 0.15 \
  --ig-threshold 0.10
```

### 4. 複数実験の比較レポート生成
```bash
python scripts/experiments/exp.py generate-report my_experiment_session \
  baseline_test \
  high_sensitivity_test \
  low_sensitivity_test
```

## 🎯 高度な実験設定

### 実験設定テンプレート作成
```bash
# 既存テンプレートをベースに設定作成
python scripts/experiments/exp.py create-config my_config standard \
  --custom '{"episodes": 750, "ged_threshold": 0.12}'
```

### パラメータスイープ実行
```bash
# 設定ファイルベースのパラメータスイープ
python scripts/experiments/exp.py run-sweep session_id sweep_base my_config \
  --sweep-params '{"ged_threshold": [0.10, 0.15, 0.20], "ig_threshold": [0.05, 0.10, 0.15]}'
```

## 📊 結果の確認

### HTMLレポートの確認
実験完了後、自動生成されたHTMLレポートを確認：

```bash
# 最新のレポートを開く
open outputs/reports/*/report.html

# 特定のセッションの結果を確認
ls outputs/[SESSION_ID]/
```

### 結果ファイル構造
```
experiments/outputs/[SESSION_ID]/[EXPERIMENT_NAME]/
├── 01_input_episodes.csv      # 入力エピソード
├── 02_insights.csv           # 検出された洞察
├── 03_experiment_logs.csv    # 詳細実験ログ
├── 04_topk_analysis.csv      # TopK類似度分析
├── 05_memory_snapshots.csv   # メモリ状態スナップショット
└── 06_experiment_results.json # 実験結果サマリー
```

## 🔄 実験のベストプラクティス

### 1. 実験前の準備
```bash
# 1. データ状態確認
python scripts/experiments/exp.py status

# 2. データ整合性チェック
python scripts/experiments/exp.py check

# 3. 必要に応じてクリーンアップ
python scripts/experiments/exp.py clean

# 4. バックアップ作成
python scripts/experiments/exp.py backup before_new_experiment
```

### 2. 同一条件での比較実験
```bash
# 同じシード値を使用して再現性を確保
python scripts/experiments/exp.py build-memory --episodes 50 --seed 42
python scripts/experiments/exp.py run session1 exp1 --seed 42 --episodes 500

# 条件を変更して比較実験
python scripts/experiments/exp.py clean
python scripts/experiments/exp.py build-memory --episodes 50 --seed 42
python scripts/experiments/exp.py run session1 exp2 --seed 42 --episodes 500 --ged-threshold 0.10
```

### 3. 実験結果の体系的な比較
```bash
# 複数実験の比較レポート生成
python scripts/experiments/exp.py generate-report session1 exp1 exp2 exp3 \
  --report-name parameter_comparison
```

## 🎛️ 利用可能なパラメータ

### 実験パラメータ
- `--episodes`: エピソード数 (デフォルト: 500)
- `--seed`: ランダムシード (デフォルト: 42)
- `--memory-dim`: メモリ次元 (デフォルト: 384)
- `--topk`: TopK近傍数 (デフォルト: 10)
- `--ged-threshold`: GED閾値 (デフォルト: 0.15)
- `--ig-threshold`: IG閾値 (デフォルト: 0.10)
- `--similarity-threshold`: 類似度閾値 (デフォルト: 0.3)

### 初期メモリパラメータ
- `--episodes`: 初期エピソード数 (デフォルト: 50)
- `--seed`: 初期メモリ生成シード (デフォルト: 42)

## 🐛 トラブルシューティング

### よくある問題と解決方法

1. **"エピソードテーブルが見つかりません"エラー**
   ```bash
   python scripts/experiments/exp.py clean
   python scripts/experiments/exp.py build-memory --episodes 50
   ```

2. **メモリ不足エラー**
   ```bash
   # エピソード数を減らす
   python scripts/experiments/exp.py run session exp --episodes 200
   ```

3. **実験結果が見つからない**
   ```bash
   # セッション確認
   ls experiments/outputs/
   
   # 特定のセッション内容確認
   ls experiments/outputs/[SESSION_ID]/
   ```

4. **パフォーマンスが悪い**
   ```bash
   # quick_testテンプレートを使用
   python scripts/experiments/exp.py workflow quick_comparison
   ```

## 📈 実験結果の解釈

### 主要メトリクス
- **洞察検出率 (insight_rate)**: 総エピソードに対する洞察数の割合
- **処理速度 (episodes_per_second)**: 1秒あたりの処理エピソード数
- **GED値 (delta_ged)**: グローバル編集距離（類似度の逆数）
- **IG値 (delta_ig)**: 情報ゲイン

### 比較分析のポイント
1. **洞察検出率**: 高すぎる場合は閾値が低すぎる可能性
2. **処理速度**: 一貫性のある速度であることを確認
3. **パラメータの影響**: 複数実験の結果から最適な設定を特定
4. **初期メモリの影響**: 0エピソードスタートとの比較

## 🎯 推奨実験シーケンス

### 初回実験（デバッグ・検証）
```bash
python scripts/experiments/exp.py workflow quick_comparison
```

### 本格的な比較実験
```bash
python scripts/experiments/exp.py workflow parameter_sensitivity
```

### 詳細分析
```bash
python scripts/experiments/exp.py workflow memory_initialization_study
```

### 最終評価
```bash
python scripts/experiments/exp.py workflow comprehensive_evaluation
```

このシステムにより、rigorous で reproducible な実験が可能になり、InsightSpike-AIの性能を正確に評価・比較できます。
