#!/usr/bin/env python3
"""
クイックテスト（10×10迷路、500ステップ）
"""

import numpy as np
import time
from datetime import datetime
import sys
import os

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent import PureMemoryAgent


def quick_test():
    """10×10迷路でクイックテスト"""
    
    print("="*60)
    print("QUICK TEST - 10×10 Maze")
    print("Pure Memory-Based Navigation")
    print("="*60)
    
    # 迷路生成（実際は11×11になる）
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(11, 11), seed=42)
    
    # エージェント作成（軽い設定）
    agent = PureMemoryAgent(
        maze=maze,
        datastore_path="../results/quick_test",
        config={
            'max_depth': 3,  # 浅い深度
            'search_k': 15   # 少ない検索数
        }
    )
    
    print(f"Start: {agent.position}, Goal: {agent.goal}")
    print(f"Max steps: 500")
    print("-" * 40)
    
    # 実験実行
    start_time = time.time()
    max_steps = 500
    
    for step in range(max_steps):
        if agent.is_goal_reached():
            elapsed = time.time() - start_time
            stats = agent.get_statistics()
            
            print(f"\n✅ SUCCESS in {step} steps!")
            print(f"   Time: {elapsed:.2f} seconds")
            print(f"   Wall hits: {stats['wall_hits']} ({stats['wall_hits']/step*100:.1f}%)")
            print(f"   Episodes: {stats['total_episodes']}")
            print(f"   Path length: {stats['path_length']}")
            
            # 深度使用
            print("\nDepth usage:")
            for depth, count in stats['depth_usage'].items():
                if count > 0:
                    print(f"  {depth}-hop: {count} times")
            
            return True
        
        # 行動
        action = agent.get_action()
        agent.execute_action(action)
        
        # 軽い進捗報告
        if step % 50 == 0 and step > 0:
            stats = agent.get_statistics()
            print(f"Step {step}: dist={stats['distance_to_goal']}, "
                  f"wall_hits={stats['wall_hits']}")
    
    # 失敗
    elapsed = time.time() - start_time
    stats = agent.get_statistics()
    
    print(f"\n❌ Failed after {max_steps} steps")
    print(f"   Time: {elapsed:.2f} seconds")
    print(f"   Final distance: {stats['distance_to_goal']}")
    print(f"   Wall hits: {stats['wall_hits']}")
    print(f"   Episodes: {stats['total_episodes']}")
    
    return False


if __name__ == "__main__":
    success = quick_test()
    
    print("\n" + "="*60)
    if success:
        print("🎉 Pure memory navigation works!")
        print("   Ready for larger experiments")
    else:
        print("📊 Need more optimization")
    print("="*60)