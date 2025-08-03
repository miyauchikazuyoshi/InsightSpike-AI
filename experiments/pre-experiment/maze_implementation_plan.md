# 迷路実験の実装計画

## Phase 1: 最小実装（1週間）

### 1.1 シンプルな2D迷路環境
```python
# experiments/pre-experiment/maze_env.py
class SimpleMaze:
    def __init__(self, size=(20, 20)):
        self.size = size
        self.grid = self.generate_maze()
        self.agent_pos = (1, 1)  # Start
        self.goal_pos = (18, 18)  # Goal
        
    def step(self, action):
        # 0:上, 1:右, 2:下, 3:左
        new_pos = self.get_new_position(action)
        
        if self.is_wall(new_pos):
            # 壁にぶつかった！
            return self.agent_pos, -1, False, {"hit_wall": True}
        else:
            self.agent_pos = new_pos
            done = (self.agent_pos == self.goal_pos)
            reward = 10 if done else -0.1
            return self.agent_pos, reward, done, {}
```

### 1.2 特徴ベースのメモリシステム
```python
# experiments/pre-experiment/maze_memory.py
class MazeMemory:
    def __init__(self):
        self.nodes = {}  # position -> node
        self.embedder = SimpleEmbedder()
        
    def should_create_node(self, obs):
        """重要な場所かどうか判定"""
        return any([
            obs['num_paths'] >= 3,     # 交差点
            obs['num_paths'] == 1,     # 行き止まり
            obs['hit_wall'],           # 壁
            obs['is_goal']             # ゴール
        ])
    
    def add_node(self, position, features):
        """ノード追加（エネルギーコスト考慮）"""
        node = {
            'position': position,
            'features': features,
            'vector': self.embedder.encode(features),
            'creation_cost': 1.0  # 高いコスト
        }
        self.nodes[position] = node
```

### 1.3 geDIGナビゲーター
```python
# experiments/pre-experiment/gediq_navigator.py
class GeDIGNavigator:
    def __init__(self, memory):
        self.memory = memory
        self.sphere_search = SphereSearch()
        
    def decide_action(self, current_pos, observation):
        # 1. 現在位置から球面探索
        nearby = self.sphere_search.search_sphere(
            current_pos, radius=5
        )
        
        # 2. 各方向の価値を評価
        action_values = {}
        for action in range(4):
            next_pos = self.get_next_pos(current_pos, action)
            
            # 壁チェック（ドーナツ探索）
            walls_ahead = self.count_walls_ahead(next_pos, nearby)
            
            # エネルギー計算
            energy = self.calculate_energy(
                next_pos, nearby, walls_ahead
            )
            
            action_values[action] = -energy  # 低エネルギーが良い
            
        return max(action_values, key=action_values.get)
```

## Phase 2: 実験設定（3日）

### 2.1 比較ベースライン
```python
# experiments/pre-experiment/baselines.py

class RandomAgent:
    """ランダム探索"""
    def decide_action(self, obs):
        return random.choice([0, 1, 2, 3])

class DFSAgent:
    """深さ優先探索（全セル記憶）"""
    def __init__(self):
        self.visited = set()
        self.stack = []

class AStarAgent:
    """A*（完全な地図前提）"""
    def __init__(self, maze_map):
        self.map = maze_map
```

### 2.2 評価指標
```python
metrics = {
    'steps_to_goal': [],      # ゴールまでのステップ数
    'memory_usage': [],       # 作成したノード数
    'wall_hits': [],          # 壁衝突回数
    'exploration_rate': [],   # 探索済みセルの割合
    'path_optimality': []     # 最短経路との比率
}
```

### 2.3 実験スクリプト
```python
# experiments/pre-experiment/run_maze_experiment.py

def run_experiment():
    # 設定
    config = {
        'maze_size': (20, 20),
        'num_episodes': 100,
        'sleep_interval': 10,  # 10エピソードごとにSleep
        'algorithms': ['random', 'dfs', 'astar', 'gediq']
    }
    
    results = {}
    
    for algo in config['algorithms']:
        agent = create_agent(algo)
        
        for episode in range(config['num_episodes']):
            maze = SimpleMaze(config['maze_size'])
            steps, nodes = run_episode(maze, agent)
            
            # geDIGの場合、定期的にSleep
            if algo == 'gediq' and episode % config['sleep_interval'] == 0:
                agent.sleep_phase()  # メモリ最適化
                
        results[algo] = agent.get_metrics()
    
    return results
```

## Phase 3: 可視化（2日）

### 3.1 リアルタイム可視化
```python
# experiments/pre-experiment/maze_visualizer.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class MazeVisualizer:
    def __init__(self, maze):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.maze = maze
        
    def draw_frame(self, agent_pos, memory_nodes):
        self.ax.clear()
        
        # 迷路描画
        self.draw_maze()
        
        # メモリノード描画
        for pos, node in memory_nodes.items():
            color = self.get_node_color(node['features'])
            self.ax.scatter(*pos, c=color, s=100)
        
        # エージェント描画
        self.ax.scatter(*agent_pos, c='red', s=200, marker='*')
        
    def save_animation(self, frames, filename):
        """探索過程をGIFとして保存"""
        ani = animation.FuncAnimation(
            self.fig, self.update, frames=frames
        )
        ani.save(filename, writer='pillow')
```

### 3.2 結果の可視化
```python
def plot_results(results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 学習曲線
    axes[0,0].set_title('Steps to Goal')
    for algo, data in results.items():
        axes[0,0].plot(data['steps'], label=algo)
    
    # 2. メモリ使用量
    axes[0,1].set_title('Memory Usage')
    axes[0,1].bar(results.keys(), 
                  [r['final_nodes'] for r in results.values()])
    
    # 3. 壁衝突回数
    axes[1,0].set_title('Wall Hits')
    for algo, data in results.items():
        axes[1,0].plot(data['wall_hits'], label=algo)
    
    # 4. 最終パス比較
    axes[1,1].set_title('Final Path Optimality')
    # ...
```

## 実験の段階的実行

### Week 1: 基本実装
1. **Day 1-2**: 迷路環境とメモリシステム
2. **Day 3-4**: geDIGナビゲーター実装
3. **Day 5**: デバッグと基本動作確認

### Week 2: 実験と評価
1. **Day 1**: ベースライン実装
2. **Day 2-3**: 比較実験実行
3. **Day 4**: 可視化システム
4. **Day 5**: 結果分析・論文用図表作成

## 期待される結果

```python
expected_results = {
    'gediq_advantages': {
        'memory_efficiency': '90% reduction vs DFS',
        'wall_avoidance': 'Learn to avoid walls after 10-20 episodes',
        'path_optimization': 'Discover shortcuts during sleep',
        'transfer_learning': 'Adapt to maze variations quickly'
    },
    
    'key_demonstrations': [
        'Autonomous map formation without explicit instructions',
        'Natural gradient emergence from energy minimization',
        'Sleep phase discovers optimal paths',
        'Feature-based memory enables generalization'
    ]
}
```

## 実装のポイント

1. **シンプルに始める**
   - 最初は20×20の迷路で十分
   - 複雑な3D表示は後回し

2. **核心機能に集中**
   - 特徴ベースメモリ
   - ドーナツ探索
   - エネルギー最小化

3. **定量的評価**
   - 明確な指標で比較
   - 統計的有意性を確認

4. **視覚的説得力**
   - メモリ形成過程のアニメーション
   - 学習曲線の明確な差

これで論文の実験セクションに十分なデータが取れるはずです！