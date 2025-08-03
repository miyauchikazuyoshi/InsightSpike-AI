#!/usr/bin/env python3
"""エピソード記憶ナビゲーターのデバッグ"""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze
from insightspike.maze_experimental.maze_config import MazeNavigatorConfig
from episodic_gedig_navigator import EpisodicGeDIGNavigator


def debug_navigation():
    """ナビゲーションの詳細デバッグ"""
    config = MazeNavigatorConfig()
    
    # 小さい迷路でテスト
    np.random.seed(42)
    maze = SimpleMaze(size=(5, 5), maze_type='dfs')
    navigator = EpisodicGeDIGNavigator(config)
    
    print(f"迷路サイズ: {maze.size}")
    print(f"スタート: {maze.start_pos}")
    print(f"ゴール: {maze.goal_pos}")
    print("-" * 40)
    
    obs = maze.reset()
    
    for step in range(50):
        print(f"\nステップ {step}:")
        print(f"  現在位置: {obs.position}")
        print(f"  可能な行動: {obs.possible_moves}")
        print(f"  ゴールか?: {obs.is_goal}")
        
        if obs.is_goal:
            print("  🎯 ゴールに到達！")
        
        old_pos = obs.position
        action = navigator.decide_action(obs, maze)
        print(f"  選択した行動: {action} ({['上', '右', '下', '左'][action]})")
        
        obs, reward, done, info = maze.step(action)
        new_pos = obs.position
        
        navigator.update_after_move(old_pos, new_pos, action)
        
        print(f"  移動後の位置: {new_pos}")
        print(f"  報酬: {reward}")
        print(f"  終了?: {done}")
        
        if done:
            print(f"\n終了！最終位置: {maze.agent_pos}, ゴール: {maze.goal_pos}")
            print(f"成功?: {maze.agent_pos == maze.goal_pos}")
            break
    
    # 探索統計
    stats = navigator.get_exploration_statistics()
    print("\n探索統計:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
        
    # エピソード記憶の詳細
    print("\nエピソード記憶の詳細:")
    for pos, memory in navigator.position_memories.items():
        print(f"\n位置 {pos}:")
        print(f"  訪問回数: {memory.visits}")
        for action in range(4):
            stats = memory.get_action_statistics(action)
            if stats['count'] > 0:
                print(f"  {['上', '右', '下', '左'][action]}: "
                      f"試行{stats['count']}回, "
                      f"成功率{stats['success_rate']:.0%}, "
                      f"平均進捗{stats['avg_progress']:.1f}")


if __name__ == "__main__":
    debug_navigation()