#!/usr/bin/env python3
"""
Compare Different Mazes
=======================

異なる迷路での視覚記憶ナビゲーションを比較
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import random
from datetime import datetime

# パスを追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from test_visual_memory_maze import VisualMemoryNavigator, generate_complex_maze


class CompactNavigator(VisualMemoryNavigator):
    """コンパクトな結果記録用ナビゲーター"""
    
    def __init__(self, maze_size: int = 30):
        super().__init__(maze_size)
        self.path_history = []
        
    def execute_action(self, action: str) -> Dict:
        result = super().execute_action(action)
        self.path_history.append(self.position)
        return result


def run_maze_experiment(seed: int, maze_size: int = 30, max_steps: int = 3000) -> Dict:
    """指定シードで迷路実験を実行"""
    print(f"\n--- Running with seed {seed} ---")
    
    # シード設定
    random.seed(seed)
    np.random.seed(seed)
    
    # ナビゲーター作成
    navigator = CompactNavigator(maze_size)
    
    # 迷路を解く
    result = navigator.solve_maze(max_steps)
    
    # 結果をまとめる
    return {
        'seed': seed,
        'success': result['success'],
        'steps': result['steps'],
        'unique_positions': result['unique_positions'],
        'episodes': result['total_episodes'],
        'efficiency': result['efficiency'],
        'path_history': navigator.path_history[::10],  # 間引き
        'maze_array': navigator.maze_env.grid,
        'goal_pos': navigator.maze_env.goal_pos
    }


def visualize_comparison(results: List[Dict]):
    """複数の結果を比較可視化"""
    
    n_results = len(results)
    fig, axes = plt.subplots(2, n_results, figsize=(6*n_results, 12))
    
    # 各結果を表示
    for i, result in enumerate(results):
        # 上段：迷路と経路
        ax1 = axes[0, i] if n_results > 1 else axes[0]
        ax1.imshow(result['maze_array'], cmap='binary', alpha=0.8)
        
        # 経路を描画
        if result['path_history']:
            path = np.array(result['path_history'])
            ax1.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, alpha=0.7)
        
        # スタートとゴール
        ax1.plot(0, 0, 'go', markersize=10, label='Start')
        gx, gy = result['goal_pos']
        ax1.plot(gx, gy, 'r*', markersize=15, label='Goal')
        
        # 成功/失敗を色で表示
        if result['success']:
            ax1.set_title(f"Seed {result['seed']}: SUCCESS\n"
                         f"Steps: {result['steps']}, Efficiency: {result['efficiency']:.1f}%",
                         color='green')
        else:
            ax1.set_title(f"Seed {result['seed']}: FAILED\n"
                         f"Steps: {result['steps']}, Explored: {result['unique_positions']}",
                         color='red')
        
        ax1.set_xlim(-1, result['maze_array'].shape[1])
        ax1.set_ylim(-1, result['maze_array'].shape[0])
        ax1.set_aspect('equal')
        ax1.legend(loc='upper right')
        
        # 下段：メトリクス比較
        ax2 = axes[1, i] if n_results > 1 else axes[1]
        metrics = ['Steps', 'Unique\nPositions', 'Episodes', 'Efficiency\n(%)']
        values = [
            result['steps'],
            result['unique_positions'],
            result['episodes'],
            result['efficiency']
        ]
        
        bars = ax2.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
        ax2.set_title(f"Metrics for Seed {result['seed']}")
        ax2.set_ylabel('Value')
        
        # 値をバーの上に表示
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(value)}' if value > 1 else f'{value:.1f}',
                    ha='center', va='bottom')
    
    plt.suptitle('Visual Memory Navigation on Different Maze Structures', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/maze_comparison_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison saved to: {filename}")
    return filename


def main():
    """メイン実行"""
    print("="*60)
    print("Comparing Visual Memory Navigation on Different Mazes")
    print("="*60)
    
    # 異なるシードで実験
    seeds = [42, 123, 456]
    results = []
    
    for seed in seeds:
        result = run_maze_experiment(seed, maze_size=30, max_steps=3000)
        results.append(result)
        
        print(f"\nSeed {seed} Results:")
        print(f"  Success: {result['success']}")
        print(f"  Steps: {result['steps']}")
        print(f"  Unique positions: {result['unique_positions']}")
        print(f"  Efficiency: {result['efficiency']:.1f}%")
    
    # 比較可視化
    visualize_comparison(results)
    
    # サマリー統計
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    success_count = sum(1 for r in results if r['success'])
    print(f"Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.0f}%)")
    
    if success_count > 0:
        avg_steps = np.mean([r['steps'] for r in results if r['success']])
        avg_efficiency = np.mean([r['efficiency'] for r in results if r['success']])
        print(f"Average steps (successful): {avg_steps:.0f}")
        print(f"Average efficiency (successful): {avg_efficiency:.1f}%")
    
    print("="*60)


if __name__ == "__main__":
    main()