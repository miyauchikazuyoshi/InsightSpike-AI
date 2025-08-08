#!/usr/bin/env python3
"""
7×7迷路でゴール指向クエリをテスト
"""

import numpy as np
import time
import sys
import os

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent_adaptive import PureMemoryAgentAdaptive
from pure_memory_agent_goal_oriented import PureMemoryAgentGoalOriented


def test_7x7_goal_oriented():
    """7×7迷路でクエリ戦略をテスト"""
    
    print("="*60)
    print("🎯 ゴール指向クエリ実験（7×7迷路）")
    print("="*60)
    
    # 7×7迷路生成
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(7, 7), seed=42)
    
    print("\n迷路:")
    for row in maze:
        print(' '.join(['.' if x == 0 else '█' for x in row]))
    
    max_steps = 100
    
    # ゴール指向エージェント
    print("\n" + "-"*60)
    print("ゴール指向クエリ（訪問=0、ゴール=1.0）")
    print("-"*60)
    
    agent = PureMemoryAgentGoalOriented(
        maze=maze,
        datastore_path="../results/7x7_goal_test",
        config={
            'max_depth': 3,
            'search_k': 10,
            'gedig_improvement_threshold': 0.05
        }
    )
    
    print(f"スタート: {agent.position}")
    print(f"ゴール: {agent.goal}")
    
    # 実行
    path = []
    for step in range(max_steps):
        if agent.is_goal_reached():
            print(f"\n✅ 成功！ {step}ステップで到達")
            break
        
        # 行動決定の詳細（最初の20ステップ）
        if step < 20:
            pos = agent.position
            action = agent.get_action()
            success = agent.execute_action(action)
            
            symbol = "→" if action == "right" else "←" if action == "left" else "↑" if action == "up" else "↓"
            result = "○" if success else "×"
            
            print(f"Step {step:2d}: {pos} {symbol} {result}")
            path.append((pos, action, success))
        else:
            action = agent.get_action()
            agent.execute_action(action)
        
        if step % 20 == 0 and step > 0:
            stats = agent.get_statistics()
            print(f"\n進捗: 距離={stats['distance_to_goal']}, "
                  f"壁衝突率={stats['wall_hits']/step*100:.1f}%")
    else:
        print(f"\n❌ {max_steps}ステップで未到達")
    
    # 統計
    stats = agent.get_statistics()
    print("\n" + "="*60)
    print("📊 最終統計")
    print("="*60)
    print(f"最終距離: {stats['distance_to_goal']}")
    print(f"壁衝突率: {stats['wall_hits']/max(step,1)*100:.1f}%")
    print(f"総エピソード: {stats['total_episodes']}")
    
    # クエリタイプ使用状況
    qt = stats.get('query_types', {})
    if qt:
        total = sum(qt.values())
        print(f"\nクエリタイプ:")
        print(f"  ゴール指向: {qt.get('goal_oriented', 0)} ({qt.get('goal_oriented', 0)/total*100:.1f}%)")
        print(f"  探索: {qt.get('exploration', 0)} ({qt.get('exploration', 0)/total*100:.1f}%)")
    
    # 深度使用
    print(f"\n深度使用:")
    for depth, count in stats['depth_usage'].items():
        if count > 0:
            print(f"  {depth}ホップ: {count}回")
    
    return agent.is_goal_reached()


if __name__ == "__main__":
    success = test_7x7_goal_oriented()
    
    print("\n" + "="*60)
    if success:
        print("🎉 ゴール指向クエリで成功！")
        print("   訪問=0、ゴール=1.0の設定が有効")
    else:
        print("📊 さらなる調整が必要")
    print("="*60)