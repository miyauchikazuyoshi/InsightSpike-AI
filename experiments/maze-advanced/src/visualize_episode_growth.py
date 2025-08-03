#!/usr/bin/env python3
"""
Episode Graph Growth Visualization
==================================

エピソードグラフの成長過程を可視化
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
import networkx as nx
from typing import List, Dict, Tuple, Set
import json
from datetime import datetime
import time
import random
from collections import defaultdict
import gc

# パスを追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from test_visual_memory_maze import VisualMemoryNavigator, Episode7D, generate_complex_maze

try:
    from insightspike.environments.maze import SimpleMaze
except ImportError:
    from src.insightspike.environments.maze import SimpleMaze


class VisualMemoryNavigatorWithTracking(VisualMemoryNavigator):
    """グラフ成長を追跡する拡張版ナビゲーター"""
    
    def __init__(self, maze_size: int = 50):
        super().__init__(maze_size)
        
        # グラフ成長の追跡
        self.graph_snapshots = []
        self.snapshot_interval = 100  # 100ステップごとにスナップショット
        self.episode_connections = defaultdict(set)  # エピソード間の接続
        
    def _record_snapshot(self):
        """現在のグラフ状態をスナップショット"""
        snapshot = {
            'step': self.step_count,
            'position': self.position,
            'episodes': len(self.episodes),
            'unique_positions': len(self.unique_positions),
            'episode_data': [
                {
                    'x': ep.x,
                    'y': ep.y,
                    'type': self._get_episode_type(ep),
                    'visit_count': ep.visit_count
                }
                for ep in self.episodes
            ],
            'connections': dict(self.episode_connections)
        }
        self.graph_snapshots.append(snapshot)
    
    def _get_episode_type(self, episode: Episode7D) -> str:
        """エピソードのタイプを判定"""
        if episode.goal_or_not:
            return 'goal'
        elif episode.wall_or_path == 'wall':
            return 'wall'
        elif episode.direction is not None:
            return 'movement'
        else:
            return 'visual'
    
    def _update_connections(self, query_episode: Episode7D, related_episodes: List[Tuple[Episode7D, float]]):
        """エピソード間の接続を更新"""
        if query_episode in self.episodes:
            query_idx = self.episodes.index(query_episode)
            for related_ep, score in related_episodes[:3]:  # 上位3つ
                if related_ep in self.episodes and score > 0.5:
                    related_idx = self.episodes.index(related_ep)
                    self.episode_connections[query_idx].add(related_idx)
                    self.episode_connections[related_idx].add(query_idx)
    
    def decide_action(self) -> str:
        """行動決定（接続追跡付き）"""
        x, y = self.position
        
        # クエリを生成して検索
        queries = self._create_queries()
        search_results = self._search_episodes(queries)
        
        # 接続を記録
        for query in queries:
            self._update_connections(query, search_results)
        
        # 元の決定ロジック
        return super().decide_action()
    
    def execute_action(self, action: str) -> Dict:
        """行動実行（スナップショット付き）"""
        result = super().execute_action(action)
        
        # 定期的にスナップショット
        if self.step_count % self.snapshot_interval == 0:
            self._record_snapshot()
        
        return result
    
    def solve_maze_with_tracking(self, max_steps: int = 10000) -> Dict:
        """追跡機能付きで迷路を解く"""
        # 初期スナップショット
        self.setup_maze()
        self._record_snapshot()
        
        # 通常の解法を実行
        result = self.solve_maze(max_steps)
        
        # 最終スナップショット
        self._record_snapshot()
        
        return result


def create_growth_visualization(snapshots: List[Dict], maze_array: np.ndarray, 
                              output_prefix: str = "episode_growth"):
    """エピソードグラフの成長を可視化"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Episode Graph Growth Process', fontsize=16, fontweight='bold')
    
    # 選択するスナップショット（6つ）
    total_snapshots = len(snapshots)
    indices = [
        0,  # 初期状態
        int(total_snapshots * 0.2),
        int(total_snapshots * 0.4),
        int(total_snapshots * 0.6),
        int(total_snapshots * 0.8),
        total_snapshots - 1  # 最終状態
    ]
    
    for idx, (ax, snap_idx) in enumerate(zip(axes.flat, indices)):
        snapshot = snapshots[snap_idx]
        
        # 背景の迷路
        ax.imshow(maze_array, cmap='binary', alpha=0.3)
        
        # エピソードをタイプ別にプロット
        episode_data = snapshot['episode_data']
        
        # タイプ別に分類
        types = defaultdict(list)
        for ep in episode_data:
            if ep['x'] is not None and ep['y'] is not None:
                types[ep['type']].append((ep['x'], ep['y'], ep['visit_count']))
        
        # タイプ別の色とサイズ
        type_styles = {
            'goal': {'color': 'red', 'marker': '*', 'size': 200},
            'wall': {'color': 'black', 'marker': 's', 'size': 30},
            'movement': {'color': 'blue', 'marker': 'o', 'size': 50},
            'visual': {'color': 'green', 'marker': '^', 'size': 40}
        }
        
        # 各タイプをプロット
        for ep_type, positions in types.items():
            if positions and ep_type in type_styles:
                style = type_styles[ep_type]
                xs, ys, visits = zip(*positions)
                
                # 訪問回数に応じてサイズを調整
                sizes = [style['size'] * (1 + min(v, 5) * 0.2) for v in visits]
                
                ax.scatter(xs, ys, c=style['color'], marker=style['marker'], 
                          s=sizes, alpha=0.7, label=ep_type)
        
        # 現在位置
        pos = snapshot['position']
        ax.plot(pos[0], pos[1], 'yo', markersize=10, markeredgecolor='black', 
                markeredgewidth=2, label='Current')
        
        # タイトルと情報
        ax.set_title(f"Step {snapshot['step']}: {snapshot['episodes']} episodes\n"
                    f"Unique positions: {snapshot['unique_positions']}")
        ax.set_xlim(-1, 50)
        ax.set_ylim(-1, 50)
        ax.set_aspect('equal')
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/{output_prefix}_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Growth visualization saved to: {filename}")
    return filename


def create_network_graph(snapshot: Dict, output_prefix: str = "episode_network"):
    """エピソード間の接続をネットワークグラフとして可視化"""
    
    if not snapshot['connections']:
        print("No connections to visualize")
        return
    
    # NetworkXグラフを構築
    G = nx.Graph()
    
    # ノードを追加
    episode_data = snapshot['episode_data']
    for i, ep in enumerate(episode_data):
        if ep['x'] is not None and ep['y'] is not None:
            G.add_node(i, pos=(ep['x'], ep['y']), type=ep['type'])
    
    # エッジを追加
    for node1, connections in snapshot['connections'].items():
        for node2 in connections:
            if node1 in G and node2 in G:
                G.add_edge(node1, node2)
    
    # 描画
    plt.figure(figsize=(12, 12))
    
    # ノードの位置
    pos = nx.get_node_attributes(G, 'pos')
    
    # ノードのタイプ別色
    node_types = nx.get_node_attributes(G, 'type')
    color_map = {
        'goal': 'red',
        'wall': 'gray',
        'movement': 'lightblue',
        'visual': 'lightgreen'
    }
    node_colors = [color_map.get(node_types.get(node, 'visual'), 'yellow') 
                   for node in G.nodes()]
    
    # グラフを描画
    nx.draw(G, pos, node_color=node_colors, node_size=100, 
            with_labels=False, edge_color='gray', alpha=0.6, width=0.5)
    
    plt.title(f'Episode Connection Network (Step {snapshot["step"]})\n'
             f'{len(G.nodes())} nodes, {len(G.edges())} edges')
    plt.axis('equal')
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/{output_prefix}_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Network graph saved to: {filename}")
    return filename


def create_animated_growth(snapshots: List[Dict], maze_array: np.ndarray,
                          output_prefix: str = "episode_growth_animation"):
    """成長過程のアニメーションを作成"""
    
    # 簡易版：主要な指標の時系列変化
    steps = [s['step'] for s in snapshots]
    episodes = [s['episodes'] for s in snapshots]
    unique_pos = [s['unique_positions'] for s in snapshots]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # エピソード数の成長
    ax1.plot(steps, episodes, 'b-', linewidth=2, label='Total Episodes')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Number of Episodes')
    ax1.set_title('Episode Growth Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # ユニーク位置の成長
    ax2.plot(steps, unique_pos, 'g-', linewidth=2, label='Unique Positions')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Number of Unique Positions')
    ax2.set_title('Exploration Coverage Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 効率性の追加（右軸）
    ax2_twin = ax2.twinx()
    efficiency = [up / (s + 1) * 100 for s, up in zip(steps, unique_pos)]
    ax2_twin.plot(steps, efficiency, 'r--', linewidth=1.5, alpha=0.7, label='Efficiency %')
    ax2_twin.set_ylabel('Efficiency (%)', color='red')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/{output_prefix}_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Growth animation saved to: {filename}")
    return filename


def main():
    """メイン実行"""
    print("="*60)
    print("Episode Graph Growth Visualization")
    print("="*60)
    
    # 乱数シードを設定
    random.seed(42)
    np.random.seed(42)
    
    # 追跡機能付きナビゲーターで実験
    navigator = VisualMemoryNavigatorWithTracking(maze_size=50)
    
    print("Running maze with tracking...")
    result = navigator.solve_maze_with_tracking(max_steps=10000)
    
    print(f"\nCollected {len(navigator.graph_snapshots)} snapshots")
    
    # 迷路配列を取得
    maze_array = navigator.maze_env.grid
    
    # 各種可視化を生成
    print("\nGenerating visualizations...")
    
    # 1. 成長過程の6段階表示
    create_growth_visualization(navigator.graph_snapshots, maze_array)
    
    # 2. 時系列グラフ
    create_animated_growth(navigator.graph_snapshots, maze_array)
    
    # 3. 最終状態のネットワークグラフ
    if navigator.graph_snapshots:
        create_network_graph(navigator.graph_snapshots[-1])
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == "__main__":
    main()