# Maze Mode実装計画（Config駆動＋ビジュアライゼーション）

## 概要

Configで`maze_mode`を指定することで、迷路解法専用の処理モードに切り替え、Layer4でリアルタイムビジュアライゼーションを行う。

## アーキテクチャ

```
Config (maze_mode: true)
    ↓
MainAgent
    ↓
Layer4 (LLM Interface) ← ここでビジュアライゼーション
    ├─ 迷路の進行状況を可視化
    └─ 知識グラフの構築過程を表示
```

## 背景

### 変更の動機
- 迷路解法は固定タスクなので、専用モードが効率的
- ビジュアライゼーションで学習過程を可視化したい
- Layer4で迷路進行とグラフ構築を同時表示

### 設計方針
- **Config駆動**: `maze_mode: true`で切り替え
- **固定クエリ**: 迷路図を初期入力、以降は自動進行
- **リアルタイム可視化**: matplotlib/networkxでアニメーション

## 実装計画

### Phase 1: Config拡張とMaze Mode定義（1日）

#### 1.1 設定モデルの拡張
```python
# config/models.py
class MazeModeConfig(BaseModel):
    enabled: bool = False
    maze_size: Tuple[int, int] = (20, 20)
    visualize: bool = True
    save_animation: bool = True
    animation_interval: int = 500  # ms
    
class InsightSpikeConfig(BaseModel):
    # 既存フィールド...
    maze_mode: MazeModeConfig = MazeModeConfig()
```

#### 1.2 迷路入力フォーマット
```python
# 迷路は2D配列として入力
maze_query = {
    "type": "maze",
    "maze": [
        [0, 0, 1, 0, 0],  # 0: 通路, 1: 壁
        [0, 1, 1, 0, 0],  # S: スタート
        [0, 0, 0, 0, 1],  # G: ゴール
        [1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ],
    "start": (0, 0),
    "goal": (4, 4)
}
```

### Phase 2: Layer4でのビジュアライゼーション実装（3日）

#### 2.1 MazeVisualizer クラス
```python
class MazeVisualizer:
    def __init__(self, maze, start, goal):
        self.maze = maze
        self.fig, (self.ax_maze, self.ax_graph) = plt.subplots(1, 2)
        self.path = [start]
        self.knowledge_graph = nx.Graph()
        
    def update_frame(self, current_pos, graph_state):
        # 左側: 迷路の進行状況
        self.draw_maze(current_pos)
        
        # 右側: 知識グラフの成長
        self.draw_knowledge_graph(graph_state)
        
    def save_animation(self, filename='maze_solving.gif'):
        # アニメーションGIF保存
```

#### 2.2 Layer4の拡張
```python
class Layer4LLMInterface:
    def __init__(self, config):
        self.config = config
        if config.maze_mode.enabled:
            self.visualizer = None
            
    def process_maze_mode(self, maze_query):
        # 初期化
        self.visualizer = MazeVisualizer(
            maze_query['maze'],
            maze_query['start'],
            maze_query['goal']
        )
        
        # 固定クエリで自動進行
        while not self.at_goal():
            # 1. 現在位置から次の行動を決定
            action = self.decide_action()
            
            # 2. 行動実行と結果取得
            result = self.execute_action(action)
            
            # 3. ビジュアライゼーション更新
            self.visualizer.update_frame(
                self.current_pos,
                self.knowledge_graph
            )
            
            # 4. 知識グラフ更新
            self.update_knowledge_graph(result)
```

### Phase 3: MainAgentの統合（1日）

#### 3.1 MainAgentの拡張
```python
class MainAgent:
    def __init__(self, config, datastore):
        self.config = config
        self.maze_mode = config.maze_mode.enabled
        
    def process_question(self, query):
        if self.maze_mode and self._is_maze_query(query):
            # Maze Mode: Layer4で特殊処理
            return self.layer4.process_maze_mode(query)
        else:
            # Language Mode: 通常処理
            return self._process_language_query(query)
```

#### 3.2 Wake Modeとの連携
```python
if self.maze_mode:
    # Maze Mode: 常にWake Mode（効率重視）
    self.processing_mode = ProcessingMode.WAKE
    # 5次元ベクトル使用
    self.vector_dim = 5
```

### Phase 4: ビジュアライゼーション詳細設計（2日）

#### 4.1 表示要素

**迷路側（左パネル）**:
- 現在位置（青い丸）
- 訪問済み経路（緑の線）
- 未訪問エリア（グレー）
- 壁（黒）
- ゴール（赤い星）

**グラフ側（右パネル）**:
- ノード: 訪問した位置
- エッジ: 移動経路
- 色分け: geDIG値でヒートマップ
- アニメーション: 新規ノード/エッジの追加

#### 4.2 実装例
```python
def visualize_maze_solving(self):
    # リアルタイム更新
    ani = FuncAnimation(
        self.fig,
        self.update_frame,
        frames=self.max_steps,
        interval=self.config.animation_interval,
        repeat=False
    )
    
    if self.config.save_animation:
        ani.save('maze_solving.gif', writer='pillow')
    
    plt.show()
```

### Phase 5: 最適化とチューニング（1日）

#### 5.1 パラメータ調整
- タイプ別の重みパラメータ
- 検出閾値の最適化
- ベクトル次元数の微調整

#### 5.2 キャッシング戦略
- タイプ別キャッシュ
- 頻出パターンの事前計算
- メモリ管理の最適化

## 期待される成果

### 1. Maze Modeの利点
- **可視化**: 学習過程が一目瞭然
- **効率**: 固定タスクに特化した処理
- **教育的価値**: geDIG理論の動作を視覚的に理解

### 2. 技術的成果
- **リアルタイムビジュアライゼーション**: matplotlib統合
- **知識グラフの成長過程**: NetworkX可視化
- **5次元ベクトルの実証**: 迷路タスクでの有効性

### 3. 研究への貢献
- Wake Modeの効果を視覚的に実証
- 知識グラフ構築過程の理解
- 効率的な探索戦略の可視化

## リスクと対策

### リスク1: タイプ誤判定
**対策**: 
- 信頼度スコアの導入
- ユーザーによる明示的指定オプション
- フォールバック機構

### リスク2: ハイブリッドクエリの複雑性
**対策**:
- 段階的処理（まず主要タイプを判定）
- 適応的重み付け
- 実験による最適な統合方法の発見

### リスク3: 既存システムとの互換性
**対策**:
- 後方互換性の維持
- 段階的移行パス
- 十分なテストカバレッジ

## 実装手順

### Step 1: Config拡張
1. `MazeModeConfig`クラス追加
2. `InsightSpikeConfig`に統合
3. デフォルト値設定

### Step 2: Layer4拡張
1. `MazeVisualizer`クラス作成
2. `process_maze_mode`メソッド追加
3. matplotlib/networkx統合

### Step 3: MainAgent統合
1. maze_mode分岐追加
2. Wake Mode固定設定
3. 5次元ベクトル使用

### Step 4: テストとデモ
1. サンプル迷路作成
2. アニメーション生成
3. 性能測定

## 次のアクション

1. **Config拡張の実装**から開始
2. **MazeVisualizerプロトタイプ**作成
3. **統合テスト**の準備