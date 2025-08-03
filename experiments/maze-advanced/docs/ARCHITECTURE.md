# Maze Agent Integration アーキテクチャ

## システム構成図

```
┌─────────────────────────────────────────────────────────────┐
│                    MazeAgentWrapper                         │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   Maze      │  │   Vector     │  │                 │  │
│  │Environment  │  │  Adapter     │  │   Visualizer    │  │
│  │             │  │              │  │                 │  │
│  │ - maze grid │  │ 384D → 5D    │  │ - maze panel   │  │
│  │ - position  │  │ 5D → 384D    │  │ - graph panel  │  │
│  │ - actions   │  │              │  │ - animation    │  │
│  └──────┬──────┘  └──────┬───────┘  └────────┬────────┘  │
│         │                │                     │           │
│         └────────────────┴─────────────────────┘           │
│                          │                                  │
│  ┌───────────────────────┴─────────────────────────────┐  │
│  │                    MainAgent                         │  │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐   │  │
│  │  │Layer 1 │  │Layer 2 │  │Layer 3 │  │Layer 4 │   │  │
│  │  │Error   │  │Memory  │  │Graph   │  │LLM     │   │  │
│  │  │Monitor │  │Manager │  │Reasoner│  │Interface│  │  │
│  │  └────────┘  └────────┘  └────────┘  └────────┘   │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## データフロー

```
1. 迷路クエリ入力
   ↓
2. MazeEnvironmentで迷路初期化
   ↓
3. MainAgentに現在状態をクエリ
   ↓
4. VectorAdapterで次元変換（384D ↔ 5D）
   ↓
5. Wake Modeで行動決定（geDIG最小化）
   ↓
6. 行動実行と結果取得
   ↓
7. 知識グラフ更新
   ↓
8. Visualizerで表示更新
   ↓
9. ゴール到達まで3-8を繰り返し
```

## コンポーネント詳細

### 1. MazeAgentWrapper
**責務**: 全体の調整と制御
```python
class MazeAgentWrapper:
    def __init__(self, config):
        self.main_agent = MainAgent(config, datastore)
        self.environment = MazeEnvironment()
        self.vector_adapter = VectorAdapter()
        self.visualizer = MazeVisualizer()
        
    def solve_maze(self, maze_query):
        # メインループ
        while not self.at_goal():
            state = self.get_current_state()
            action = self.decide_action(state)
            result = self.execute_action(action)
            self.update_visualization()
```

### 2. MazeEnvironment
**責務**: 迷路の状態管理
```python
class MazeEnvironment:
    def __init__(self, maze, start, goal):
        self.maze = maze
        self.current_pos = start
        self.goal = goal
        self.visited = set()
        
    def get_possible_actions(self):
        # 現在位置から可能な行動
        
    def execute_action(self, action):
        # 行動実行と結果返却
```

### 3. VectorAdapter
**責務**: ベクトル次元の相互変換
```python
class VectorAdapter:
    def to_compact(self, high_dim_vector):
        # 384次元 → 5次元
        # [X, Y, 方向, 結果, 訪問回数]
        
    def to_semantic(self, compact_vector):
        # 5次元 → 384次元
        # 位置情報を自然言語的に表現
```

### 4. MazeVisualizer
**責務**: リアルタイムビジュアライゼーション
```python
class MazeVisualizer:
    def __init__(self):
        self.fig, (self.ax_maze, self.ax_graph) = plt.subplots(1, 2)
        
    def update(self, maze_state, graph_state):
        self.draw_maze(maze_state)
        self.draw_knowledge_graph(graph_state)
        
    def save_animation(self, filename):
        # GIFとして保存
```

## Wake Mode統合

### geDIG最小化戦略
```python
# MazeAgentWrapper内で
def decide_action(self, state):
    # Wake Mode固定
    self.main_agent.processing_mode = ProcessingMode.WAKE
    
    # 既知パターンとのマッチング
    if self.has_similar_pattern(state):
        return self.apply_known_pattern(state)
    
    # 効率的な探索
    return self.efficient_exploration(state)
```

### 5次元ベクトル表現
```python
def create_compact_vector(self, state):
    return np.array([
        state.x / self.maze_width,      # 正規化X座標
        state.y / self.maze_height,     # 正規化Y座標
        state.last_action / 4.0,        # 方向 (0-3を正規化)
        state.last_result,              # 結果 (-1: 壁, 0: 通路, 1: ゴール)
        min(state.visits / 10.0, 1.0)  # 訪問回数（上限付き正規化）
    ])
```

## 将来的なLayer5統合

### 現在（実験）
```
MainAgent
└─ MazeAgentWrapper（外部ラッパー）
   └─ Visualizer
```

### 将来（本実装）
```
MainAgent
├─ Layer1: Error Monitor
├─ Layer2: Memory Manager
├─ Layer3: Graph Reasoner
├─ Layer4: LLM Interface
└─ Layer5: Visualization（NEW!）
   ├─ TaskVisualizer（抽象クラス）
   ├─ MazeVisualizer
   └─ FutureVisualizer
```

## パフォーマンス最適化

### メモリ効率
- 5次元ベクトル: 5 * 4 bytes = 20 bytes
- 384次元ベクトル: 384 * 4 bytes = 1,536 bytes
- **削減率**: 98.7%（1/77）

### 処理速度
- ベクトル演算: 77倍高速化（理論値）
- 類似度計算: O(5) vs O(384)
- メモリアクセス: キャッシュ効率向上

## 拡張性

### 新しいタスクへの適用
```python
class TaskAgentWrapper(ABC):
    @abstractmethod
    def create_compact_vector(self, state):
        pass
        
    @abstractmethod
    def visualize(self, state):
        pass

class MazeAgentWrapper(TaskAgentWrapper):
    # 迷路特化実装

class FutureTaskWrapper(TaskAgentWrapper):
    # 他タスク用実装
```

これにより、実験の成功後スムーズに本実装へ移行可能。