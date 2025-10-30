# Maze Experimental Module

このディレクトリには迷路ナビゲーション実験で使用した試作実装が含まれています。

## ⚠️ 重要な注意事項

これらのファイルは実験的な実装であり、本番環境での使用は推奨されません。

## 移動されたファイル（2025-08-03）

### 1. navigators/ (from src/insightspike/navigators/)
- simple_action_navigator.py
- structured_action_navigator.py
- action_memory_navigator.py
- blind_experience_navigator.py
- experience_memory_navigator.py
- passage_graph_navigator.py
- wall_graph_navigator.py
- wall_only_gediq_navigator.py
- wall_aware_gediq_navigator.py
- pure_gediq_navigator.py
- simple_gediq_navigator.py
- gediq_navigator.py

### 2. query/ (from src/insightspike/query/)
- dimension_aware_sphere_search.py
- logarithmic_sphere_search.py
- wake_mode_searcher.py

### 3. algorithms/ (from src/insightspike/algorithms/)
- vector_space_gedig.py
- gedig_gradient_field.py

### 4. config/ (from src/insightspike/config/)
- maze_config.py

### 5. environments/ (copied from src/insightspike/environments/)
- complex_maze.py
- proper_maze_generator.py

## インポートパスの更新

既存のコードでこれらのモジュールを使用している場合は、インポートパスを更新してください：

```python
# 旧
from insightspike.navigators import ExperienceMemoryNavigator

# 新
from insightspike.maze_experimental.navigators import ExperienceMemoryNavigator
```

## 実験結果

これらの実装を使用した実験結果は以下のディレクトリに保存されています：
- experiments/pre-experiment/
- experiments/maze-sota-comparison/
- experiments/maze-agent-integration/
- experiments/episodic-message-passing/