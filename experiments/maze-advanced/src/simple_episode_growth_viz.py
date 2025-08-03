#!/usr/bin/env python3
"""
Simple Episode Growth Visualization
===================================

シンプルなエピソードグラフの成長可視化（NetworkX不要版）
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json
from datetime import datetime
import random
from collections import defaultdict

# パスを追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from test_visual_memory_maze import VisualMemoryNavigator, Episode7D, generate_complex_maze

try:
    from insightspike.environments.maze import SimpleMaze
except ImportError:
    from src.insightspike.environments.maze import SimpleMaze


class TrackingNavigator(VisualMemoryNavigator):
    """簡易追跡機能付きナビゲーター"""
    
    def __init__(self, maze_size: int = 50):
        super().__init__(maze_size)
        self.history_snapshots = []
        self.snapshot_steps = [0, 500, 1000, 2000, 4000, 8000]  # 固定ステップ
        
    def execute_action(self, action: str) -> Dict:
        """行動実行（スナップショット付き）"""
        result = super().execute_action(action)
        
        # 指定ステップでスナップショット
        if self.step_count in self.snapshot_steps:
            self._save_snapshot()
        
        return result
    
    def _save_snapshot(self):
        """現在の状態を保存"""
        snapshot = {
            'step': self.step_count,
            'position': self.position,
            'episodes': len(self.episodes),
            'unique_positions': len(self.unique_positions),
            'episode_positions': [
                (ep.x, ep.y, self._get_episode_type(ep))
                for ep in self.episodes
                if ep.x is not None and ep.y is not None
            ]
        }
        self.history_snapshots.append(snapshot)
        print(f"Snapshot saved at step {self.step_count}")
    
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


def visualize_growth_stages(navigator: TrackingNavigator, maze_array: np.ndarray):
    """成長段階を可視化"""
    
    # 初期スナップショットを追加
    navigator.history_snapshots.insert(0, {
        'step': 0,
        'position': (0, 0),
        'episodes': 1,
        'unique_positions': 1,
        'episode_positions': [(navigator.maze_env.goal_pos[0], navigator.maze_env.goal_pos[1], 'goal')]
    })
    
    # 6段階の可視化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Episode Memory Growth Process', fontsize=16, fontweight='bold')
    
    # カラーマップ
    type_colors = {
        'goal': 'red',
        'wall': 'darkgray',
        'movement': 'blue',
        'visual': 'green'
    }
    
    for idx, (ax, snapshot) in enumerate(zip(axes.flat, navigator.history_snapshots[:6])):
        # 背景の迷路
        ax.imshow(maze_array, cmap='binary', alpha=0.3)
        
        # エピソードをタイプ別にプロット
        type_groups = defaultdict(list)
        for x, y, ep_type in snapshot['episode_positions']:
            type_groups[ep_type].append((x, y))
        
        # 各タイプをプロット
        for ep_type, positions in type_groups.items():
            if positions:
                xs, ys = zip(*positions)
                color = type_colors.get(ep_type, 'gray')
                marker = {'goal': '*', 'wall': 's', 'movement': 'o', 'visual': '^'}.get(ep_type, '.')
                size = {'goal': 200, 'wall': 30, 'movement': 50, 'visual': 40}.get(ep_type, 30)
                
                ax.scatter(xs, ys, c=color, marker=marker, s=size, 
                          alpha=0.7, label=f'{ep_type} ({len(positions)})')
        
        # 現在位置
        pos = snapshot['position']
        ax.plot(pos[0], pos[1], 'yo', markersize=12, markeredgecolor='black', 
                markeredgewidth=2, label='Agent')
        
        # 情報表示
        ax.set_title(f"Step {snapshot['step']}\n"
                    f"Episodes: {snapshot['episodes']}, "
                    f"Explored: {snapshot['unique_positions']}")
        ax.set_xlim(-1, 50)
        ax.set_ylim(-1, 50)
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/episode_growth_stages_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nGrowth stages visualization saved to: {filename}")
    return filename


def visualize_metrics_timeline(navigator: TrackingNavigator):
    """メトリクスの時系列変化を可視化"""
    
    # データ収集
    steps = []
    episodes = []
    unique_positions = []
    efficiencies = []
    
    # 履歴から抽出
    for snapshot in navigator.history_snapshots:
        steps.append(snapshot['step'])
        episodes.append(snapshot['episodes'])
        unique_positions.append(snapshot['unique_positions'])
        if snapshot['step'] > 0:
            efficiencies.append(snapshot['unique_positions'] / snapshot['step'] * 100)
        else:
            efficiencies.append(0)
    
    # プロット
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    
    # エピソード数
    ax1.plot(steps, episodes, 'b-o', linewidth=2, markersize=8)
    ax1.fill_between(steps, episodes, alpha=0.3)
    ax1.set_ylabel('Total Episodes', fontsize=12)
    ax1.set_title('Episode Memory Growth Timeline', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 探索範囲
    ax2.plot(steps, unique_positions, 'g-o', linewidth=2, markersize=8)
    ax2.fill_between(steps, unique_positions, alpha=0.3, color='green')
    ax2.set_ylabel('Unique Positions Explored', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 効率性
    ax3.plot(steps, efficiencies, 'r-o', linewidth=2, markersize=8)
    ax3.fill_between(steps, efficiencies, alpha=0.3, color='red')
    ax3.set_xlabel('Steps', fontsize=12)
    ax3.set_ylabel('Efficiency (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 注釈を追加
    for i, (s, e, u) in enumerate(zip(steps, episodes, unique_positions)):
        if i % 2 == 0:  # 偶数インデックスのみ
            ax1.annotate(f'{e}', (s, e), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
            ax2.annotate(f'{u}', (s, u), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/episode_metrics_timeline_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics timeline saved to: {filename}")
    return filename


def create_heatmap_evolution(navigator: TrackingNavigator, maze_array: np.ndarray):
    """訪問頻度ヒートマップの進化"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Exploration Heatmap Evolution', fontsize=14, fontweight='bold')
    
    # 3つの時点を選択
    checkpoints = [
        (navigator.history_snapshots[1], 'Early (500 steps)'),
        (navigator.history_snapshots[3], 'Middle (2000 steps)'),
        (navigator.history_snapshots[-1], f'Final ({navigator.history_snapshots[-1]["step"]} steps)')
    ]
    
    for ax, (snapshot, title) in zip(axes, checkpoints):
        # 訪問マップを作成
        visit_map = np.zeros((50, 50))
        
        # エピソード位置から訪問頻度を推定
        for x, y, ep_type in snapshot['episode_positions']:
            if ep_type in ['movement', 'visual']:
                visit_map[y, x] += 1
        
        # マスク付き表示
        masked_visit = np.ma.masked_where(maze_array == 1, visit_map)
        im = ax.imshow(masked_visit, cmap='hot', interpolation='nearest')
        
        ax.set_title(f'{title}\nUnique: {snapshot["unique_positions"]}')
        ax.set_xlim(-0.5, 49.5)
        ax.set_ylim(-0.5, 49.5)
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/heatmap_evolution_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap evolution saved to: {filename}")
    return filename


def main():
    """メイン実行"""
    print("="*60)
    print("Simple Episode Growth Visualization")
    print("="*60)
    
    # 乱数シードを設定
    random.seed(42)
    np.random.seed(42)
    
    # 追跡機能付きナビゲーターで実験
    print("Running maze with tracking...")
    navigator = TrackingNavigator(maze_size=50)
    
    # 迷路を解く
    result = navigator.solve_maze(max_steps=8500)  # 少し余裕を持たせる
    
    if not result['success']:
        print("Warning: Maze not fully solved, but continuing with visualization...")
    
    print(f"\nCollected {len(navigator.history_snapshots)} snapshots")
    
    # 迷路配列を取得
    maze_array = navigator.maze_env.grid
    
    # 各種可視化を生成
    print("\nGenerating visualizations...")
    
    # 1. 成長段階の可視化
    visualize_growth_stages(navigator, maze_array)
    
    # 2. メトリクスの時系列
    visualize_metrics_timeline(navigator)
    
    # 3. ヒートマップの進化
    create_heatmap_evolution(navigator, maze_array)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Total steps: {navigator.step_count}")
    print(f"Total episodes: {len(navigator.episodes)}")
    print(f"Unique positions: {len(navigator.unique_positions)}")
    print(f"Success: {result['success']}")
    print(f"Efficiency: {result['efficiency']:.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()