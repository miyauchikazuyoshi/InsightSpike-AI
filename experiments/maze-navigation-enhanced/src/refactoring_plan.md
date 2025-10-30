# リファクタリング計画

## 目的
現在の一枚岩実装を責務ごとにクラス分離し、保守性・拡張性を向上させる

## 現状の問題点
1. `phase2_branch_visualization_final.py`に全機能が集中
2. 責務が不明確で変更が困難
3. テストが書きにくい
4. geDIG閾値などのパラメータ調整が困難

## クラス設計

### 1. Core Components

#### EpisodeManager（既存を改良）
```python
class EpisodeManager:
    """エピソードのライフサイクル管理"""
    
    責務:
    - エピソード作成（観測時）
    - 訪問回数の初期値設定
    - 訪問回数の更新（移動時）
    - エピソード検索
    
    メソッド:
    - observe(position, maze) -> Dict[str, Episode]
    - move(position, direction) -> bool
    - get_episode(position, direction) -> Episode
    - get_episodes_at_step(step) -> Dict
```

#### GraphManager
```python
class GraphManager:
    """エピソードグラフの構築と管理"""
    
    責務:
    - エッジ配線戦略の実装
    - グラフ構造の維持
    - グラフ履歴の記録
    
    メソッド:
    - add_episode_node(episode)
    - wire_edges(episodes, strategy='simple')
    - wire_with_gedig(episodes, threshold=0.3)  # Approach D
    - get_graph_snapshot() -> nx.Graph
```

#### VectorProcessor
```python
class VectorProcessor:
    """ベクトル処理と計算"""
    
    責務:
    - 8次元ベクトル生成
    - 重み適用
    - 距離計算
    
    メソッド:
    - create_vector(pos, direction, is_wall, visits) -> np.ndarray
    - apply_weights(vector, weights) -> np.ndarray
    - calculate_distance(vec1, vec2, weights) -> float
```

### 2. Decision Components

#### DecisionEngine
```python
class DecisionEngine:
    """行動選択ロジック"""
    
    責務:
    - クエリ生成
    - ノルム検索
    - 確率計算と選択
    
    メソッド:
    - create_query(position) -> np.ndarray
    - norm_search(query, episodes, weights) -> List[Tuple[float, Episode]]
    - select_action(episodes, temperature=0.1) -> str
```

#### GeDIGEvaluator
```python
class GeDIGEvaluator:
    """geDIG計算と評価"""
    
    責務:
    - geDIG値計算
    - グラフ短絡検出
    - 閾値判定
    
    メソッド:
    - calculate(g1: nx.Graph, g2: nx.Graph) -> float
    - detect_shortcut(g1, g2) -> bool
    - should_create_edge(gedig_value, threshold) -> bool
```

### 3. Navigation Components

#### BranchDetector
```python
class BranchDetector:
    """分岐点検出と管理"""
    
    責務:
    - 分岐進入検出
    - 分岐完了検出
    - バックトラック判定
    
    メソッド:
    - detect_branch_entry(position, maze) -> bool
    - detect_branch_completion(position) -> bool
    - should_backtrack(gedig_value, threshold) -> bool
```

#### MazeNavigator
```python
class MazeNavigator:
    """メインナビゲーションシステム"""
    
    責務:
    - コンポーネント統合
    - メインループ実行
    - イベント管理
    
    構成:
    - episode_manager: EpisodeManager
    - graph_manager: GraphManager
    - vector_processor: VectorProcessor
    - decision_engine: DecisionEngine
    - gedig_evaluator: GeDIGEvaluator
    - branch_detector: BranchDetector
    
    メソッド:
    - run(maze, start, goal, max_steps=1000)
    - step() -> bool
    - visualize()
```

## 実装順序

### Phase 1: 基盤クラス実装
1. VectorProcessor（独立性が高い）
2. GeDIGEvaluator（独立性が高い）
3. EpisodeManager改良（既存を拡張）

### Phase 2: グラフ・意思決定
4. GraphManager（エッジ配線戦略）
5. DecisionEngine（ノルム検索）
6. BranchDetector（分岐検出）

### Phase 3: 統合
7. MazeNavigator（全体統合）
8. 動作テスト
9. パラメータ調整

## テスト計画

### ユニットテスト
- 各クラスの個別機能テスト
- エッジケースの確認

### 統合テスト
- T字迷路での動作確認
- 分岐検出の精度確認
- geDIG閾値の効果検証

### パフォーマンステスト
- 25x25迷路での実行時間
- メモリ使用量
- グラフサイズの推移

## 期待される効果

1. **保守性向上**
   - 各クラスの責務が明確
   - 変更の影響範囲が限定的

2. **テスタビリティ向上**
   - 各クラスを独立してテスト可能
   - モックを使った単体テスト

3. **拡張性向上**
   - 新しいエッジ配線戦略の追加が容易
   - 異なる意思決定アルゴリズムの実装が容易

4. **パラメータ調整の容易化**
   - geDIG閾値の動的調整
   - 重みベクトルの実験的変更

## 次のステップ

1. このプランのレビューと承認
2. Phase 1の実装開始
3. 各クラスの詳細設計
4. 実装とテスト