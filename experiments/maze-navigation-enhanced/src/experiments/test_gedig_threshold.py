#!/usr/bin/env python3
"""
geDIG閾値による行き詰まり検知実験
"""

import numpy as np
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from navigation.maze_navigator import MazeNavigator


def create_dead_end_maze():
    """多数の袋小路を持つ迷路を作成"""
    maze = np.ones((15, 15), dtype=int)
    
    # メイン通路（縦）
    for y in range(1, 14):
        maze[y, 7] = 0
    
    # 横の通路（ゴールへ）
    for x in range(7, 14):
        maze[1, x] = 0
    
    # 袋小路1（左上）
    for x in range(1, 7):
        maze[3, x] = 0
    
    # 袋小路2（左中）
    for x in range(1, 7):
        maze[7, x] = 0
    
    # 袋小路3（左下）
    for x in range(1, 7):
        maze[11, x] = 0
    
    # 袋小路4（右側の短い枝）
    for x in range(8, 11):
        maze[5, x] = 0
    
    # 袋小路5（下側の短い枝）
    for y in range(9, 12):
        maze[y, 9] = 0
    
    return maze


@pytest.mark.skip("Experimental benchmark script – skip in unit test run")
def test_threshold(maze=None, start_pos=None, goal_pos=None, threshold=-0.2, max_steps=300):
    """特定の閾値でテスト"""
    
    weights = np.array([
        1.0, 1.0, 0.0, 0.0, 3.0, 2.0, 0.1, 0.0
    ])
    
    navigator = MazeNavigator(
        maze=maze,
        start_pos=start_pos,
        goal_pos=goal_pos,
        weights=weights,
        temperature=0.1,
        gedig_threshold=0.3,
        backtrack_threshold=threshold,  # テスト対象の閾値
        wiring_strategy='simple'
    )
    
    # ナビゲーション実行
    success = navigator.run(max_steps=max_steps)
    
    # 結果収集
    stats = navigator.get_statistics()
    
    # バックトラックイベントの数を数える
    backtrack_count = sum(
        1 for event in navigator.event_log
        if event['type'] == 'backtrack_trigger'
    )
    
    # 袋小路への進入回数（分岐完了の回数）
    dead_end_count = stats['branch_stats']['completed_branches']
    
    result = {
        'threshold': threshold,
        'success': success,
        'steps': navigator.step_count,
        'path_length': len(navigator.path),
        'unique_positions': stats['unique_positions'],
        'backtrack_triggers': backtrack_count,
        'dead_ends_explored': dead_end_count,
        'gedig_history': navigator.gedig_history
    }
    
    return result


def main():
    print("="*60)
    print("geDIG THRESHOLD EXPERIMENT FOR DEADLOCK DETECTION")
    print("="*60)
    
    # 迷路作成
    maze = create_dead_end_maze()
    start_pos = (7, 13)
    goal_pos = (13, 1)
    
    print("\n実験迷路（◎:袋小路）:")
    h, w = maze.shape
    for y in range(h):
        row = ""
        for x in range(w):
            if (x, y) == start_pos:
                row += "S "
            elif (x, y) == goal_pos:
                row += "G "
            elif maze[y, x] == 1:
                row += "█ "
            else:
                # 袋小路の端を検出
                neighbors = 0
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < w and 0 <= ny < h and maze[ny, nx] == 0:
                        neighbors += 1
                if neighbors == 1:
                    row += "◎ "  # 袋小路の端
                else:
                    row += "· "
        print(row)
    
    # 異なる閾値でテスト
    thresholds = [-0.5, -0.3, -0.2, -0.1, -0.05, 0.0]
    results = []
    
    print(f"\nテスト開始...")
    print("-"*60)
    
    for threshold in thresholds:
        print(f"\nTesting threshold = {threshold}...")
        result = test_threshold(maze, start_pos, goal_pos, threshold)
        results.append(result)
        
        print(f"  Success: {result['success']}")
        print(f"  Steps: {result['steps']}")
        print(f"  Backtrack triggers: {result['backtrack_triggers']}")
        print(f"  Dead ends explored: {result['dead_ends_explored']}")
    
    # 結果の分析
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    print("\n閾値別パフォーマンス:")
    print("Threshold | Success | Steps | Backtracks | Dead Ends")
    print("-"*55)
    for r in results:
        success_mark = "✓" if r['success'] else "✗"
        print(f"{r['threshold']:8.2f} | {success_mark:^7} | {r['steps']:5d} | {r['backtrack_triggers']:10d} | {r['dead_ends_explored']:9d}")
    
    # 最適な閾値の推定
    successful_results = [r for r in results if r['success']]
    if successful_results:
        # 成功した中で最も効率的な（ステップ数が少ない）閾値
        best = min(successful_results, key=lambda x: x['steps'])
        print(f"\n推奨閾値: {best['threshold']}")
        print(f"  理由: {best['steps']}ステップで成功（最小）")
        print(f"  バックトラック回数: {best['backtrack_triggers']}")
    
    # geDIG値の分布を可視化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, result in enumerate(results[:6]):
        ax = axes[i]
        if result['gedig_history']:
            ax.plot(result['gedig_history'], label=f"Threshold={result['threshold']}")
            ax.axhline(y=result['threshold'], color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.set_title(f"Threshold={result['threshold']} ({'Success' if result['success'] else 'Fail'})")
            ax.set_xlabel('Step')
            ax.set_ylabel('geDIG Value')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.suptitle('geDIG Values During Navigation with Different Thresholds')
    plt.tight_layout()
    
    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'../../results/gedig_threshold/experiment_{timestamp}.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"\nグラフを保存: {output_path}")
    plt.show()
    
    # 結果をテキストで保存
    result_file = f'../../results/gedig_threshold/results_{timestamp}.txt'
    with open(result_file, 'w') as f:
        f.write("geDIG Threshold Experiment Results\n")
        f.write("="*60 + "\n\n")
        
        for r in results:
            f.write(f"Threshold: {r['threshold']}\n")
            f.write(f"  Success: {r['success']}\n")
            f.write(f"  Steps: {r['steps']}\n")
            f.write(f"  Backtrack triggers: {r['backtrack_triggers']}\n")
            f.write(f"  Dead ends explored: {r['dead_ends_explored']}\n")
            f.write("-"*40 + "\n")
        
        if successful_results:
            f.write(f"\nRecommended threshold: {best['threshold']}\n")
    
    print(f"結果を保存: {result_file}")


if __name__ == "__main__":
    main()