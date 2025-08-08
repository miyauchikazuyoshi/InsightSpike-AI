#!/usr/bin/env python3
"""
小さい迷路でのテスト実験
"""

import numpy as np
import sys
import os

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent import PureMemoryAgent


def test_small_maze():
    """5×5の小さい迷路でテスト"""
    
    print("="*60)
    print("SMALL MAZE TEST (5×5)")
    print("Pure Movement Episodic Memory")
    print("="*60)
    
    # 小さい迷路を生成
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(5, 5), seed=42)
    
    print("Maze:")
    for row in maze:
        print(' '.join(['.' if x == 0 else '#' for x in row]))
    
    # エージェント作成
    agent = PureMemoryAgent(
        maze=maze,
        datastore_path="../results/test_5x5",
        config={'max_depth': 3, 'search_k': 10}
    )
    
    print(f"\nStart: {agent.position}, Goal: {agent.goal}")
    print("-" * 40)
    
    # ナビゲート
    max_steps = 100
    for step in range(max_steps):
        if agent.is_goal_reached():
            print(f"\n✅ SUCCESS in {step} steps!")
            break
        
        # 行動
        action = agent.get_action()
        success = agent.execute_action(action)
        
        # 進捗
        if step % 10 == 0:
            stats = agent.get_statistics()
            print(f"Step {step}: pos={stats['position']}, "
                  f"dist={stats['distance_to_goal']}, "
                  f"wall_hits={stats['wall_hits']}")
    else:
        print(f"\n❌ Failed after {max_steps} steps")
    
    # 最終統計
    stats = agent.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Episodes: {stats['total_episodes']}")
    print(f"  Wall hits: {stats['wall_hits']}")
    print(f"  Path length: {stats['path_length']}")
    print(f"  Distance to goal: {stats['distance_to_goal']}")


if __name__ == "__main__":
    test_small_maze()