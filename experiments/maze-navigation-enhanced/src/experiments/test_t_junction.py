#!/usr/bin/env python3
"""
T字迷路での動作テスト
"""

import numpy as np
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from navigation.maze_navigator import MazeNavigator


def create_t_junction_maze():
    """T字型迷路を作成"""
    maze = np.ones((11, 11), dtype=int)
    
    # 縦の通路（中央）
    for y in range(1, 10):
        maze[y, 5] = 0
    
    # 横の通路（上部）
    for x in range(1, 10):
        maze[1, x] = 0
    
    # 左の行き止まり分岐
    for x in range(1, 5):
        maze[3, x] = 0
    maze[3, 5] = 0  # 接続部
    
    return maze


def print_maze_with_path(maze, path, current_pos, start_pos, goal_pos):
    """迷路とパスを表示"""
    h, w = maze.shape
    
    # パスを集合に変換
    path_set = set(path)
    
    for y in range(h):
        row = ""
        for x in range(w):
            pos = (x, y)
            
            if pos == current_pos:
                row += "◎ "  # 現在位置
            elif pos == start_pos:
                row += "S "  # スタート
            elif pos == goal_pos:
                row += "G "  # ゴール
            elif pos in path_set:
                row += "○ "  # 通過済み
            elif maze[y, x] == 1:
                row += "█ "  # 壁
            else:
                row += "· "  # 通路
        print(row)


def main():
    print("="*60)
    print("T-JUNCTION MAZE NAVIGATION TEST")
    print("="*60)
    
    # 迷路作成
    maze = create_t_junction_maze()
    start_pos = (5, 9)
    goal_pos = (9, 1)
    
    # デフォルト重み
    weights = np.array([
        1.0, 1.0, 0.0, 0.0, 3.0, 2.0, 0.1, 0.0
    ])
    
    print("\n初期迷路:")
    print_maze_with_path(maze, [], start_pos, start_pos, goal_pos)
    
    # ナビゲーター作成
    navigator = MazeNavigator(
        maze=maze,
        start_pos=start_pos,
        goal_pos=goal_pos,
        weights=weights,
        temperature=0.1,
        gedig_threshold=0.3,
        backtrack_threshold=-0.2,
        wiring_strategy='simple'
    )
    
    print(f"\nStart: {start_pos}, Goal: {goal_pos}")
    print("\nStarting navigation...")
    print("-"*60)
    
    # ナビゲーション実行
    success = navigator.run(max_steps=200)
    
    # 結果表示
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    if success:
        print(f"✅ Goal reached in {navigator.step_count} steps!")
    else:
        print(f"❌ Failed to reach goal after {navigator.step_count} steps")
    
    # 最終状態の迷路表示
    print("\n最終状態:")
    print_maze_with_path(
        maze,
        navigator.path,
        navigator.current_pos,
        start_pos,
        goal_pos
    )
    
    # 統計表示
    stats = navigator.get_statistics()
    
    print("\n統計情報:")
    print(f"  総ステップ数: {stats['steps']}")
    print(f"  パス長: {stats['path_length']}")
    print(f"  ユニーク位置数: {stats['unique_positions']}")
    print(f"  総エピソード数: {stats['episode_stats']['total_episodes']}")
    print(f"  グラフノード数: {stats['graph_stats']['num_nodes']}")
    print(f"  グラフエッジ数: {stats['graph_stats']['num_edges']}")
    
    if 'gedig_stats' in stats:
        print(f"\ngeDIG統計:")
        print(f"  平均: {stats['gedig_stats']['mean']:.4f}")
        print(f"  標準偏差: {stats['gedig_stats']['std']:.4f}")
        print(f"  最小: {stats['gedig_stats']['min']:.4f}")
        print(f"  最大: {stats['gedig_stats']['max']:.4f}")
    
    # 分岐統計
    branch_stats = stats['branch_stats']
    print(f"\n分岐統計:")
    print(f"  分岐点数: {branch_stats['total_branch_points']}")
    print(f"  完了分岐数: {branch_stats['completed_branches']}")
    
    # イベントログ表示
    print(f"\n主要イベント:")
    for event in navigator.event_log[:10]:  # 最初の10個
        print(f"  Step {event['step']}: {event['type']} - {event['message']}")
    
    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"../../results/t_junction/test_{timestamp}.txt"
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    
    with open(result_file, 'w') as f:
        f.write(f"T-Junction Maze Test - {timestamp}\n")
        f.write(f"Success: {success}\n")
        f.write(f"Steps: {navigator.step_count}\n")
        f.write(f"Path length: {len(navigator.path)}\n")
        f.write(f"Unique positions: {stats['unique_positions']}\n")
    
    print(f"\n結果を保存: {result_file}")


if __name__ == "__main__":
    main()