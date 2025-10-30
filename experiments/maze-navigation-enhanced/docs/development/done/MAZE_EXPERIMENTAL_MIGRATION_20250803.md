# Maze Experimental Migration Report
Date: 2025-08-03

## 概要
`src/insightspike`ディレクトリから迷路実験関連の試作ファイルを`src/insightspike/maze_experimental`に移動しました。

## 移動されたファイル

### 1. ディレクトリ移動
- `src/insightspike/navigators/` → `src/insightspike/maze_experimental/navigators/`
  - 12個のナビゲーターファイル

### 2. ファイル移動
- `src/insightspike/query/` → `src/insightspike/maze_experimental/query/`
  - dimension_aware_sphere_search.py
  - logarithmic_sphere_search.py  
  - wake_mode_searcher.py

- `src/insightspike/config/maze_config.py` → `src/insightspike/maze_experimental/maze_config.py`

- `src/insightspike/algorithms/` → `src/insightspike/maze_experimental/algorithms/`
  - vector_space_gedig.py
  - gedig_gradient_field.py

### 3. コピーされたファイル（オリジナルは残存）
- `src/insightspike/environments/complex_maze.py` → `src/insightspike/maze_experimental/environments/complex_maze.py`
- `src/insightspike/environments/proper_maze_generator.py` → `src/insightspike/maze_experimental/environments/proper_maze_generator.py`

## インポートパスの更新

### 自動更新されたファイル数
- experiments/内の63ファイル
- navigators/内の12ファイル
- 合計: 75ファイル

### 更新例
```python
# 旧
from insightspike.navigators import ExperienceMemoryNavigator

# 新  
from insightspike.maze_experimental.navigators import ExperienceMemoryNavigator
```

## ディレクトリ構造

```
src/insightspike/maze_experimental/
├── __init__.py
├── README.md
├── maze_config.py
├── navigators/
│   ├── __init__.py
│   ├── simple_action_navigator.py
│   ├── structured_action_navigator.py
│   ├── action_memory_navigator.py
│   ├── blind_experience_navigator.py
│   ├── experience_memory_navigator.py
│   ├── passage_graph_navigator.py
│   ├── wall_graph_navigator.py
│   ├── wall_only_gediq_navigator.py
│   ├── wall_aware_gediq_navigator.py
│   ├── pure_gediq_navigator.py
│   ├── simple_gediq_navigator.py
│   └── gediq_navigator.py
├── query/
│   ├── dimension_aware_sphere_search.py
│   ├── logarithmic_sphere_search.py
│   └── wake_mode_searcher.py
├── algorithms/
│   ├── vector_space_gedig.py
│   └── gedig_gradient_field.py
└── environments/
    ├── complex_maze.py
    └── proper_maze_generator.py
```

## 今後の推奨事項

1. **新規実験での使用**
   - 新しい実験では`maze_experimental`からインポートしてください
   - これらは実験的実装なので本番環境では使用しないでください

2. **残存ファイルの確認**
   - `src/insightspike/environments/`にまだ迷路関連ファイルが残っている可能性があります
   - 必要に応じて追加の整理を検討してください

3. **ドキュメント更新**
   - 各実験のREADMEでインポートパスの変更を記載することを推奨します

## 備考
- queryディレクトリは完全に空になったため、削除可能です
- wake_mode関連のファイル（gedig_wake_mode.py、wake_sleep_config.py）は本体に残しました（テストやデモで使用されているため）