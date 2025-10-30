# Navigator Files Usage Status

このディレクトリには迷路実験の試作ナビゲーターが含まれています。

## ファイル一覧と使用状況

### 1. simple_action_navigator.py
- **説明**: シンプルなアクション記憶ナビゲーター
- **使用状況**: experiments/pre-experiment/test_simple_action.py で使用
- **結論**: ✅ 使用されている（実験で活用）

### 2. structured_action_navigator.py  
- **説明**: 構造化されたアクションナビゲーター
- **使用状況**: experiments/pre-experiment/test_structured_action.py で使用
- **結論**: ✅ 使用されている（実験で活用）

### 3. action_memory_navigator.py
- **説明**: アクションメモリを持つナビゲーター
- **使用状況**: experiments/pre-experiment/test_action_memory.py で使用
- **結論**: ✅ 使用されている（実験で活用）

### 4. blind_experience_navigator.py
- **説明**: 視覚情報なしの経験ベースナビゲーター
- **使用状況**: experiments/pre-experiment/test_blind_navigator.py で使用
- **結論**: ✅ 使用されている（実験で活用）

### 5. experience_memory_navigator.py
- **説明**: 経験記憶を持つナビゲーター
- **使用状況**: experiments/pre-experiment/test_experience_memory.py で使用
- **結論**: ✅ 使用されている（実験で活用）

### 6. passage_graph_navigator.py
- **説明**: 通路グラフナビゲーター
- **使用状況**: experiments/pre-experiment/test_passage_graph.py で使用
- **結論**: ✅ 使用されている（実験で活用）

### 7. wall_graph_navigator.py
- **説明**: 壁グラフナビゲーター
- **使用状況**: experiments/pre-experiment/test_wall_graph.py で使用
- **結論**: ✅ 使用されている（実験で活用）

### 8. wall_only_gediq_navigator.py
- **説明**: 壁情報のみのGeDIQナビゲーター
- **使用状況**: experiments/pre-experiment/test_wall_only.py で使用
- **結論**: ✅ 使用されている（実験で活用）

### 9. wall_aware_gediq_navigator.py
- **説明**: 壁認識GeDIQナビゲーター
- **使用状況**: experiments/pre-experiment/test_dfs_maze.py で使用
- **結論**: ✅ 使用されている（実験で活用）

### 10. pure_gediq_navigator.py
- **説明**: 純粋なGeDIQナビゲーター
- **使用状況**: experiments/pre-experiment/visualize_maze_solution.py で使用
- **結論**: ✅ 使用されている（実験で活用）

### 11. simple_gediq_navigator.py
- **説明**: シンプルなGeDIQナビゲーター
- **使用状況**: experiments/pre-experiment/run_maze_experiment.py で使用
- **結論**: ✅ 使用されている（実験で活用）

### 12. gediq_navigator.py
- **説明**: GeDIQナビゲーター（基本実装）
- **使用状況**: experiments/pre-experiment/run_maze_experiment.py で使用
- **結論**: ✅ 使用されている（実験で活用）

## まとめ

すべてのナビゲーターファイルは実験で使用されています。これらは迷路実験の異なるアプローチを試すための試作品として作成され、実際に実験で活用されています。

## 推奨事項

1. これらのファイルは実験的な実装なので、本番環境では使用しないでください
2. 必要に応じて `experimental/` または `prototypes/` ディレクトリに移動することを検討
3. 各ファイルにドキュメントを追加して、どの実験で使用されたか明記すると良いでしょう