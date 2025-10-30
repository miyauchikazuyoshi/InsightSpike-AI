# リファクタリング計画（ディレクトリ構造付き）

## 新しいディレクトリ構造

```
experiments/maze-navigation-enhanced/
├── src/
│   ├── core/                      # コアコンポーネント（再利用可能）
│   │   ├── __init__.py
│   │   ├── episode_manager.py     # エピソード管理
│   │   ├── vector_processor.py    # ベクトル処理
│   │   ├── graph_manager.py       # グラフ管理
│   │   └── gedig_evaluator.py     # geDIG計算
│   │
│   ├── navigation/                # ナビゲーション関連
│   │   ├── __init__.py
│   │   ├── decision_engine.py     # 行動選択
│   │   ├── branch_detector.py     # 分岐検出
│   │   └── maze_navigator.py      # メインナビゲーター
│   │
│   ├── experiments/                # 実験用スクリプト
│   │   ├── test_t_junction.py     # T字迷路テスト
│   │   ├── test_25x25_maze.py     # 大規模迷路テスト
│   │   ├── test_gedig_threshold.py # geDIG閾値実験
│   │   └── benchmark.py           # パフォーマンス測定
│   │
│   ├── visualization/              # 可視化ツール
│   │   ├── __init__.py
│   │   ├── maze_visualizer.py     # 迷路可視化
│   │   ├── graph_visualizer.py    # グラフ可視化
│   │   └── metrics_plotter.py     # メトリクス表示
│   │
│   ├── utils/                      # ユーティリティ
│   │   ├── __init__.py
│   │   ├── maze_generator.py      # 迷路生成
│   │   └── config.py              # 設定管理
│   │
│   └── legacy/                     # 旧実装（参照用）
│       ├── phase2_branch_visualization_final.py
│       └── episode_manager.py
│
├── tests/                          # ユニットテスト
│   ├── core/
│   │   ├── test_episode_manager.py
│   │   ├── test_vector_processor.py
│   │   ├── test_graph_manager.py
│   │   └── test_gedig_evaluator.py
│   │
│   └── navigation/
│       ├── test_decision_engine.py
│       ├── test_branch_detector.py
│       └── test_maze_navigator.py
│
├── configs/                        # 設定ファイル
│   ├── default.yaml               # デフォルト設定
│   ├── experiment.yaml            # 実験用設定
│   └── weights.yaml               # 重みベクトル設定
│
├── results/                        # 実験結果
│   ├── t_junction/
│   ├── 25x25_maze/
│   └── gedig_threshold/
│
└── docs/                          # ドキュメント
    ├── architecture.md            # アーキテクチャ説明
    ├── usage.md                   # 使い方
    └── api.md                     # API仕様

```

## ディレクトリ別の役割

### 1. `core/` - 再利用可能なコアコンポーネント
- エピソード管理、ベクトル処理、グラフ管理など
- 他のプロジェクトでも使える汎用的な実装
- 依存関係を最小限に

### 2. `navigation/` - 迷路ナビゲーション固有の実装
- 意思決定、分岐検出など
- 迷路探索に特化した機能

### 3. `experiments/` - 実験スクリプト
- **一時的な実験はここに配置**
- 実行可能なスクリプトのみ
- 各実験は独立して実行可能

### 4. `visualization/` - 可視化ツール
- 結果の表示・分析用
- matplotlib/networkxを使用

### 5. `utils/` - 共通ユーティリティ
- 迷路生成、設定管理など
- 複数のモジュールから使用される機能

### 6. `legacy/` - 旧実装の保管
- リファクタリング前のコード
- 参照用に残すが、新実装では使用しない

## 実装フェーズ（更新版）

### Phase 1: ディレクトリ構造とコアクラス
1. ディレクトリ構造の作成
2. `core/` モジュールの実装
   - vector_processor.py
   - gedig_evaluator.py
   - episode_manager.py（既存を改良）
   - graph_manager.py

### Phase 2: ナビゲーションコンポーネント
3. `navigation/` モジュールの実装
   - decision_engine.py
   - branch_detector.py
   - maze_navigator.py

### Phase 3: ユーティリティと可視化
4. `utils/` モジュールの実装
   - maze_generator.py
   - config.py
5. `visualization/` モジュールの基本実装

### Phase 4: テストと実験
6. ユニットテストの作成
7. 統合テスト（`experiments/test_t_junction.py`）
8. パフォーマンステスト

## ファイル整理計画

### 移動するファイル
```bash
# 旧実装をlegacyへ
mv phase2_branch_visualization_final.py legacy/
mv episode_manager.py legacy/

# 実験スクリプトを整理
# phase2_*.py → experiments/に適切な名前で配置
```

### 削除対象
- phase2_*.py の大部分（必要な部分は新実装に統合）
- debug_*.py
- test_*.py（新しいテストに置き換え）

## 利点

1. **明確な責務分離**
   - どこに何があるか一目瞭然
   - 実験スクリプトと本実装が分離

2. **再利用性向上**
   - coreモジュールは他プロジェクトでも使用可能
   - 適切なインポートパスで管理

3. **実験の独立性**
   - experiments/内のスクリプトは独立実行可能
   - 一時的な実験と恒久的な実装が区別される

4. **保守性向上**
   - テストが整理されている
   - ドキュメントと実装が対応

## 次のステップ

1. ディレクトリ構造の作成
2. 既存コードの移動・整理
3. Phase 1の実装開始