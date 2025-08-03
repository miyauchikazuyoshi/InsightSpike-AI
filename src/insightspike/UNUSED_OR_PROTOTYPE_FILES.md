# 未使用または試作段階のファイル一覧

## 1. navigators/ ディレクトリ（迷路実験の試作品）
すべてのナビゲーターファイルは実験で使用されていますが、本番環境では使用されていません：
- `simple_action_navigator.py` - シンプルなアクション記憶
- `structured_action_navigator.py` - 構造化アクション
- `action_memory_navigator.py` - アクションメモリ
- `blind_experience_navigator.py` - 視覚なし経験ベース
- `experience_memory_navigator.py` - 経験記憶
- `passage_graph_navigator.py` - 通路グラフ
- `wall_graph_navigator.py` - 壁グラフ
- `wall_only_gediq_navigator.py` - 壁のみGeDIQ
- `wall_aware_gediq_navigator.py` - 壁認識GeDIQ
- `pure_gediq_navigator.py` - 純粋GeDIQ
- `simple_gediq_navigator.py` - シンプルGeDIQ
- `gediq_navigator.py` - 基本GeDIQ

**状態**: 実験専用、本番未使用

## 2. query/ ディレクトリ（未使用の検索実装）
以下のファイルは実装されているが、どこからも参照されていない：
- `dimension_aware_sphere_search.py` - 次元認識球体検索
- `logarithmic_sphere_search.py` - 対数球体検索
- `wake_mode_searcher.py` - Wake mode検索

**状態**: 未使用（ドキュメントでのみ言及）

## 3. algorithms/ の試作実装
- `vector_space_gedig.py` - ベクトル空間でのgeDIG最適化
  - ドキュメント`hypothesis_vector_space_gedig.md`でのみ言及
  - 実際の使用例なし
  
- `gedig_gradient_field.py` - GeDIG勾配場
  - ドキュメントでのみ言及、実装はプロトタイプ

- `gedig_wake_mode.py` - Wake modeのGeDIG実装
  - examples/wake_mode_demo.pyで使用
  - tests/でテストされているが、本番未使用

- `query_type_processor.py` - クエリタイプ処理
  - maze_agent_wrapper.pyでインポートされているが、実際には使用されていない可能性

## 4. config/ の試作設定
- `maze_config.py` - 迷路設定（navigatorsと関連）
- `wake_sleep_config.py` - Wake/Sleep設定
- `message_passing_config.py` - メッセージパッシング設定

**状態**: 実験用設定ファイル

## 5. environments/ の迷路関連
- `complex_maze.py` - 複雑な迷路生成器
  - 一部の実験で使用されているが、maze.pyに統合可能
- `proper_maze_generator.py` - 適切な迷路生成器
  - maze.pyとtest_real_100_maze.pyで使用
  - maze.pyに統合済みの可能性

## 推奨アクション

### 即座に移動すべきファイル
1. `navigators/` ディレクトリ全体を `experiments/maze-prototypes/` に移動
2. `query/` の未使用ファイルを `prototypes/query/` に移動

### 統合を検討すべきファイル
1. `complex_maze.py` と `proper_maze_generator.py` を `maze.py` に統合
2. 実験用の設定ファイルを一つの `experimental_configs.py` に統合

### 削除を検討すべきファイル
1. 完全に未使用で、将来的にも使用予定がないファイル
2. 実験が終了し、結果が得られたプロトタイプ

## 整理後のディレクトリ構造案
```
src/insightspike/
├── core/           # コア機能のみ
├── algorithms/     # 本番で使用するアルゴリズム
├── implementations/# 実装
└── experimental/   # 実験的な機能
    ├── navigators/ # 迷路ナビゲーター
    ├── query/      # 実験的検索
    └── configs/    # 実験用設定
```