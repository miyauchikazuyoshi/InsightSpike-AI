# 迷路生成とビジュアライゼーションルール

## 1. 迷路生成ルール

### 1.1 基本設定
- **デフォルトサイズ**: 50×50（必ず奇数にする: 51×51）
- **アルゴリズム**: DFS（深さ優先探索）による穴掘り法
- **特性**: 完全迷路（解が必ず1つ存在）

### 1.2 実装詳細

```python
from insightspike.environments.proper_maze_generator import ProperMazeGenerator

# 50×50迷路の生成
generator = ProperMazeGenerator()
maze = generator.generate_dfs_maze(size=(51, 51), seed=42)

# 迷路の仕様
# 0 = 通路（path）
# 1 = 壁（wall）
# スタート: (1, 1)
# ゴール: (49, 49)
```

### 1.3 迷路の複雑度調整

```python
# 複雑度レベル
COMPLEXITY_LEVELS = {
    'simple': (15, 15),    # 学習確認用
    'medium': (25, 25),    # 中間評価用
    'complex': (51, 51),   # 本実験用（50×50）
    'extreme': (101, 101)  # 限界テスト用
}
```

### 1.4 迷路生成の再現性

```python
# シード固定で再現可能
EXPERIMENT_SEEDS = {
    'training': 42,
    'validation': 123,
    'test': 456
}
```

## 2. ビジュアライゼーションルール

### 2.1 基本的な可視化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_maze_navigation(maze, path, visit_counts, title="Navigation Result"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. 迷路と経路
    ax1 = axes[0]
    ax1.imshow(maze, cmap='binary', alpha=0.3)
    
    # パスを描画
    if path:
        path_array = np.array(path)
        ax1.plot(path_array[:, 1], path_array[:, 0], 
                'b-', linewidth=2, alpha=0.7)
    
    # スタートとゴール
    ax1.plot(1, 1, 'go', markersize=10, label='Start')
    ax1.plot(maze.shape[1]-2, maze.shape[0]-2, 
            'ro', markersize=10, label='Goal')
    
    ax1.set_title('Maze & Path')
    ax1.legend()
    ax1.axis('equal')
    
    # 2. 訪問ヒートマップ
    ax2 = axes[1]
    visit_map = np.zeros_like(maze, dtype=float)
    for (x, y), count in visit_counts.items():
        visit_map[x, y] = count
    
    im2 = ax2.imshow(visit_map, cmap='hot', interpolation='nearest')
    ax2.set_title('Visit Frequency Heatmap')
    plt.colorbar(im2, ax=ax2)
    
    # 3. エピソード密度
    ax3 = axes[2]
    # エピソードの位置分布を表示
    episode_density = calculate_episode_density(episodes)
    im3 = ax3.imshow(episode_density, cmap='viridis', interpolation='nearest')
    ax3.set_title('Episode Density')
    plt.colorbar(im3, ax=ax3)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig
```

### 2.2 エピソードグラフの可視化

```python
def visualize_episode_graph(index, sample_size=100):
    """エピソード間の関係をグラフとして可視化"""
    import networkx as nx
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # サンプリング（大規模な場合）
    if len(index.graph.nodes()) > sample_size:
        nodes = random.sample(list(index.graph.nodes()), sample_size)
        subgraph = index.graph.subgraph(nodes)
    else:
        subgraph = index.graph
    
    # レイアウト計算
    pos = nx.spring_layout(subgraph, k=2, iterations=50)
    
    # エッジの描画（geDIG値で色分け）
    edges = subgraph.edges(data=True)
    edge_colors = [e[2].get('gedig', 0.5) for e in edges]
    
    nx.draw_networkx_edges(subgraph, pos, 
                          edge_color=edge_colors,
                          edge_cmap=plt.cm.RdYlGn_r,
                          width=1, alpha=0.6)
    
    # ノードの描画（成功/失敗で色分け）
    node_colors = []
    for node in subgraph.nodes():
        if node < len(index.metadata):
            success = index.metadata[node].get('success', False)
            node_colors.append('green' if success else 'red')
        else:
            node_colors.append('gray')
    
    nx.draw_networkx_nodes(subgraph, pos,
                          node_color=node_colors,
                          node_size=50, alpha=0.8)
    
    ax.set_title(f'Episode Graph (geDIG connections)\n'
                f'Nodes: {len(subgraph.nodes())}, '
                f'Edges: {len(subgraph.edges())}')
    ax.axis('off')
    
    return fig
```

### 2.3 時系列分析の可視化

```python
def visualize_learning_progress(metrics_history):
    """学習の進行状況を可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    steps = range(len(metrics_history['wall_hits']))
    
    # 1. 壁衝突率の推移
    ax1 = axes[0, 0]
    wall_hit_rate = [h/max(s,1) for h, s in 
                     zip(metrics_history['wall_hits'], steps)]
    ax1.plot(steps, wall_hit_rate, 'r-', alpha=0.7)
    ax1.set_title('Wall Hit Rate Over Time')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Wall Hit Rate')
    ax1.grid(True, alpha=0.3)
    
    # 2. ゴールまでの距離
    ax2 = axes[0, 1]
    ax2.plot(steps, metrics_history['distance_to_goal'], 'b-', alpha=0.7)
    ax2.set_title('Distance to Goal')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Manhattan Distance')
    ax2.grid(True, alpha=0.3)
    
    # 3. メッセージパッシング深度の使用頻度
    ax3 = axes[1, 0]
    depth_usage = metrics_history['depth_usage']
    depths = list(depth_usage.keys())
    counts = list(depth_usage.values())
    ax3.bar(depths, counts, alpha=0.7, color='green')
    ax3.set_title('Message Passing Depth Usage')
    ax3.set_xlabel('Depth (hops)')
    ax3.set_ylabel('Usage Count')
    
    # 4. エピソード数の増加
    ax4 = axes[1, 1]
    ax4.plot(steps, metrics_history['episode_count'], 'g-', alpha=0.7)
    ax4.set_title('Episode Memory Growth')
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Total Episodes')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Learning Progress Analysis')
    plt.tight_layout()
    return fig
```

### 2.4 50×50迷路専用の可視化

```python
def visualize_large_maze_segments(maze, path, segment_size=10):
    """大規模迷路を区画ごとに可視化"""
    height, width = maze.shape
    n_segments = (height // segment_size) + 1
    
    fig, axes = plt.subplots(n_segments, n_segments, 
                            figsize=(20, 20))
    
    for i in range(n_segments):
        for j in range(n_segments):
            ax = axes[i, j]
            
            # 区画の範囲
            y_start = i * segment_size
            y_end = min((i+1) * segment_size, height)
            x_start = j * segment_size
            x_end = min((j+1) * segment_size, width)
            
            # 区画を表示
            segment = maze[y_start:y_end, x_start:x_end]
            ax.imshow(segment, cmap='binary')
            
            # この区画内のパスを表示
            segment_path = [(y-y_start, x-x_start) 
                          for y, x in path 
                          if y_start <= y < y_end and x_start <= x < x_end]
            
            if segment_path:
                segment_array = np.array(segment_path)
                ax.plot(segment_array[:, 1], segment_array[:, 0],
                       'b-', linewidth=2)
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'[{i},{j}]', fontsize=8)
    
    plt.suptitle('50×50 Maze Segmented View')
    plt.tight_layout()
    return fig
```

## 3. エピソード検索の実装

### 3.1 GeDIGAwareIntegratedIndexの使用

```python
# 既存の実装を使用
sys.path.append('../maze-optimized-search/src')
from gedig_aware_integrated_index import GeDIGAwareIntegratedIndex

# または、メインコードから
from insightspike.index import IntegratedVectorGraphIndex
```

### 3.2 検索パラメータ（50×50迷路用）

```python
INDEX_CONFIG_50x50 = {
    'similarity_threshold': 0.4,  # より多くの接続を許可
    'gedig_threshold': 0.7,       # 品質の閾値
    'gedig_weight': 0.3,          # バランスされた重み
    'max_edges_per_node': 20,    # 大規模グラフ用
    'dimension': 7                # エピソードベクトルの次元
}

# 検索パラメータ
SEARCH_PARAMS_50x50 = {
    'k': 30,                      # 検索する近傍数
    'max_depth': 5,               # メッセージパッシングの最大深度
    'decay_factor': 0.8           # 深度ごとの減衰率
}
```

## 4. 実験設定（50×50迷路）

### 4.1 パフォーマンス目標

```yaml
50x50_maze_targets:
  max_steps: 25000  # 50×50×10
  target_success_rate: 0.7  # 70%成功
  max_wall_hit_rate: 0.3    # 30%以下
  max_episodes: 50000        # メモリ制限
```

### 4.2 評価メトリクス

```python
def evaluate_50x50_performance(results):
    """50×50迷路での性能評価"""
    metrics = {
        'success_rate': sum(r['success'] for r in results) / len(results),
        'avg_steps': np.mean([r['steps'] for r in results if r['success']]),
        'avg_wall_hit_rate': np.mean([r['wall_hits']/r['steps'] for r in results]),
        'avg_episodes': np.mean([r['total_episodes'] for r in results]),
        'convergence_rate': calculate_convergence_rate(results)
    }
    
    # 目標達成の判定
    metrics['meets_targets'] = (
        metrics['success_rate'] >= 0.7 and
        metrics['avg_wall_hit_rate'] <= 0.3 and
        metrics['avg_episodes'] <= 50000
    )
    
    return metrics
```

## 5. 実装チェックリスト

- [x] ProperMazeGeneratorを使用（DFS方式）
- [x] 51×51サイズ（奇数調整済み）
- [ ] GeDIGAwareIntegratedIndexを使用
- [ ] 7次元エピソードベクトル
- [ ] 視覚エピソードの事前追加
- [ ] 5ホップメッセージパッシング
- [ ] ビジュアライゼーション関数群

## 6. 注意事項

1. **メモリ使用量**: 50×50迷路では大量のエピソードが生成されるため、適切なメモリ管理が必要
2. **計算時間**: メッセージパッシングの深度とエピソード数により指数的に増加
3. **可視化の制限**: 全エピソードグラフは巨大になるため、サンプリングが必要