# 実験ディレクトリ整理計画

## アピールすべき重要な実験

### 1. **maze-sota-comparison** ⭐️ 最重要
- **成果**: PPOに対して学習なしで大幅に勝利（成功率96.7% vs 47.0%）
- **特徴**: SOTA手法との比較、迷路難易度分析、理論的裏付け
- **保持すべきファイル**:
  - RESULTS_SUMMARY.md
  - comparison_results.png
  - ppo_training_curves.png
  - detailed_backtrack_analysis.png
  - message_passing関連の可視化

### 2. **episodic-message-passing** ⭐️ 重要
- **成果**: エピソードをノードとして扱う新しいアプローチ
- **特徴**: 異物なエントロピーの概念、メッセージパッシング
- **保持すべきファイル**:
  - EXPERIMENT_SUMMARY.md
  - results/の可視化画像
  - gedig_*_navigator.pyの最終版のみ

### 3. **question_answer** ⭐️ 重要
- **成果**: 500知識エントリ、100質問での洞察生成能力の検証
- **特徴**: 難易度別評価、クエリストレージ検証
- **保持すべきファイル**:
  - README.md
  - EXPERIMENT_DESIGN.md
  - results/visualizations/
  - 最終的な結果ファイル

### 4. **maze-agent-integration** 📝 部分的に重要
- **成果**: Visual Memory Navigation、Sleep Cycle実装
- **特徴**: 7次元エピソード、Null Goal実験
- **保持すべきファイル**:
  - RESULTS.md
  - null_goal_comparison.md
  - visual_memory_50x50_*.png（最新のみ）
  - slam_comparison_*.png

## 削除対象

### 1. **pre-experiment** 🗑️ 削除推奨
- 初期の試作実験
- navigatorの基本実装テスト
- → 成果はmaze-sota-comparisonに統合済み

### 2. **maze_navigation** 🗑️ 削除推奨
- 単純な設計ドキュメントのみ
- 実装なし

### 3. **maze_visualization** 🗑️ 削除推奨
- READMEのみで内容なし

### 4. **wake-mode** 🗑️ 削除推奨
- 結果画像のみ
- 実装は本体に統合済み

### 5. 重複ファイルの削除
- data_snapshots/の古いスナップショット（最新のみ残す）
- 各実験のsrc/内の試作ファイル（最終版のみ残す）
- 中間結果のJSON/PNG（最終結果のみ残す）

## 整理後の構造案

```
experiments/
├── maze-sota-comparison/     # PPO比較、SOTA分析
│   ├── README.md
│   ├── RESULTS_SUMMARY.md
│   └── results/             # 主要な結果画像のみ
├── episodic-message-passing/ # メッセージパッシング実験
│   ├── README.md
│   ├── EXPERIMENT_SUMMARY.md
│   └── results/
├── question-answer/          # Q&A洞察生成実験  
│   ├── README.md
│   ├── results/
│   └── data/               # 最小限のテストデータ
└── maze-advanced/           # maze-agent-integrationから改名
    ├── README.md
    ├── RESULTS.md
    └── results/            # 重要な結果のみ
```

## 実行手順

1. バックアップ確認（既に完了）
2. 削除対象ディレクトリを削除
3. 各実験内の不要ファイルを削除
4. READMEを更新して整理内容を記録