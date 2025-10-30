# 実験出力規約 (Output Convention)

## ディレクトリ構造

全ての実験は以下の統一されたディレクトリ構造に従う：

```
results/
└── {experiment_name}/
    └── {run_id}/
        ├── metrics.json      # 必須: コアメトリクス
        ├── curves.png        # 推奨: 学習曲線
        ├── graph.pkl         # オプション: グラフ構造
        ├── episodes.json     # オプション: エピソードデータ
        └── log.txt          # オプション: 実行ログ
```

## run_id形式

```
{YYYYMMDD-HHMMSS}_seed{xxxx}
```

例：
- `20250808-143025_seed42` （シード指定あり）
- `20250808-143025` （シード指定なし）

## metrics.json形式

### 必須フィールド
```json
{
  "experiment_name": "maze_11x11",
  "run_id": "20250808-143025_seed42",
  "timestamp": "2025-08-08T14:30:25.123456",
  "seed": 42,
  
  "delta_ged_mean": 0.234,
  "delta_ged_std": 0.045,
  "delta_ig_mean": 0.567,
  "delta_ig_std": 0.089,
  "avg_search_depth": 3.2,
  "search_k": 30,
  "total_episodes": 1500,
  "runtime_seconds": 45.67,
  "memory_mb": 128.5
}
```

### オプションフィールド（実験タイプ別）

#### 迷路系
```json
{
  "success_rate": 0.75,
  "avg_steps_to_goal": 234,
  "wall_hit_rate": 0.15,
  "unique_cells_visited": 89
}
```

#### QA系
```json
{
  "f1_score": 0.823,
  "precision": 0.856,
  "recall": 0.792,
  "avg_response_time_ms": 234
}
```

#### グラフ系
```json
{
  "graph_nodes": 1500,
  "graph_edges": 4567,
  "avg_node_degree": 3.04,
  "clustering_coefficient": 0.234
}
```

## curves.png形式

標準的な4パネル構成：

```
┌─────────────┬─────────────┐
│ Delta GED   │ Delta IG    │
│ (over time) │ (over time) │
├─────────────┼─────────────┤
│ Task Metric │ Memory/Time │
│ (e.g. loss) │ Usage       │
└─────────────┴─────────────┘
```

## 実装例

```python
from insightspike.experiments.core_metrics import CoreMetrics, ExperimentRunner

def run_maze_experiment(metrics: CoreMetrics):
    """迷路実験の例"""
    # 実験実行
    for step in range(100):
        # geDIG記録
        metrics.record_gedig(delta_ged=0.3, delta_ig=0.5)
        metrics.record_search(depth=3, k=30)
    
    # タスク固有メトリクス
    metrics.record_custom('success_rate', 0.8)
    metrics.record_custom('avg_steps_to_goal', 150)
    
    return "完了"

# 実行
runner = ExperimentRunner('maze_11x11', config={'seed': 42})
result = runner.run(run_maze_experiment)
```

## 既存実験の移行

### Phase 1: 新規実験から適用
- 新しく作成する実験は必ずこの規約に従う
- `ExperimentRunner`クラスを使用

### Phase 2: 重要実験の移行
以下の実験を優先的に移行：
1. `pure-movement-episodic-memory/`
2. `maze-sota-comparison/`
3. `question-answer/`

### Phase 3: 全実験の統一
- スクリプトで一括変換
- 古いファイルは`legacy/`に移動

## 比較ツール

```python
from insightspike.experiments.core_metrics import CoreMetrics

# 複数実験の比較
comparison = CoreMetrics.compare(
    'results/maze_11x11/20250808-143025_seed42/metrics.json',
    'results/maze_11x11/20250808-150000_seed43/metrics.json'
)

print(f"Delta GED: {comparison['metrics']['delta_ged_mean']}")
```

## チェックリスト

実験実装時の確認事項：

- [ ] `ExperimentRunner`を使用している
- [ ] `metrics.json`に必須フィールドが全て含まれている
- [ ] run_idが正しい形式である
- [ ] 出力ディレクトリが規約に従っている
- [ ] タイムスタンプがISO形式である
- [ ] シード値が記録されている（使用した場合）

## 可視化ダッシュボード（将来）

```bash
# 全実験の横断ビュー
python -m insightspike.experiments.dashboard

# 特定実験の詳細
python -m insightspike.experiments.dashboard --experiment maze_11x11
```

## 更新履歴

- 2025-08-08: 初版作成
- CoreMetricsクラス実装
- 出力パス規約制定