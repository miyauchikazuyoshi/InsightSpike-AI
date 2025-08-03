# 実験ディレクトリ整理報告書
Date: 2025-08-03

## 実施内容

### 削除したディレクトリ
1. **pre-experiment/** - 初期試作実験（maze-sota-comparisonに統合済み）
2. **maze_navigation/** - 設計ドキュメントのみ（実装なし）
3. **maze_visualization/** - 空のディレクトリ
4. **wake-mode/** - 結果画像のみ（実装は本体に移動済み）

### ディレクトリ名変更
- **maze-agent-integration** → **maze-advanced**

### ファイルクリーンアップ

#### maze-advanced/src/
削除したファイル：
- autonomous_maze_agent.py～v4.py（初期試作版）
- debug_*.py（デバッグ用一時ファイル）
- pure_experience_*.py（試作版）
- test_*_maze.py（個別テストファイル）
- quick_*.py（クイックテスト）

#### episodic-message-passing/src/
削除したファイル：
- simple_episode_navigator.py（初期版）
- episode_navigator.py（初期版）
- gedig_improved_navigator.py（中間版）
- gedig_compact_navigator.py（中間版）

保持した重要ファイル：
- gedig_final_navigator.py
- gedig_balanced_navigator.py
- gedig_gnn_navigator.py

#### question_answer/data_snapshots/
- 最新2つのスナップショットのみ保持
- 古いスナップショット8個を削除

## 整理後の構造

```
experiments/
├── EXPERIMENT_GUIDELINES.md
├── VISUALIZATION_ORGANIZATION_20250803.md
├── EXPERIMENT_CLEANUP_PLAN.md
├── CLEANUP_REPORT_20250803.md
│
├── maze-sota-comparison/      # ⭐️ PPO比較実験（最重要）
│   ├── RESULTS_SUMMARY.md
│   ├── comparison_results.png
│   ├── ppo_training_curves.png
│   └── [その他の重要な結果]
│
├── episodic-message-passing/  # ⭐️ メッセージパッシング実験
│   ├── EXPERIMENT_SUMMARY.md
│   ├── results/              # 可視化結果
│   └── src/                 # 最終版のみ
│
├── question_answer/          # ⭐️ Q&A洞察生成実験
│   ├── README.md
│   ├── EXPERIMENT_DESIGN.md
│   ├── results/
│   └── data_snapshots/      # 最新2つのみ
│
└── maze-advanced/           # 📝 高度な迷路実験
    ├── RESULTS.md
    ├── null_goal_comparison.md
    └── results/

```

## 成果のハイライト

### 1. maze-sota-comparison
- **PPOに対する圧倒的勝利**: 学習なしで成功率96.7%（PPO: 47.0%）
- ゼロショット学習の優位性を実証

### 2. episodic-message-passing
- エピソードをノードとして扱う革新的アプローチ
- 異物なエントロピーの概念導入
- メッセージパッシングによる知識伝播

### 3. question_answer
- 500知識エントリ×100質問での大規模検証
- 難易度別の洞察生成能力評価
- クエリストレージ機能の実証

### 4. maze-advanced
- 7次元エピソード構造
- Visual Memory Navigation
- Null Goal実験（抽象的目標の学習）

## 削減効果
- ディレクトリ数: 8 → 4（50%削減）
- ファイル数: 大幅削減（試作版・中間版を削除）
- 重要な成果は全て保持