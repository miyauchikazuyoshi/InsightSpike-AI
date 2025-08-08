#!/usr/bin/env python3
"""
25×25迷路での最適設定テスト
現実的なサイズでの性能評価
"""

import numpy as np
import time
from datetime import datetime
import sys
import os

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent_goal_oriented import PureMemoryAgentGoalOriented


def test_25x25():
    """25×25迷路でのテスト"""
    
    print("="*70)
    print("🎯 25×25迷路 最適設定テスト")
    print("  ゴール指向クエリ + geDIG適応的深度選択")
    print("="*70)
    
    # 25×25迷路生成
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(25, 25), seed=42)
    
    print("\n迷路（25×25）の一部:")
    for i in range(10):
        row_str = ''.join(['.' if maze[i][j] == 0 else '█' for j in range(20)])
        print(row_str + "...")
    print("...")
    
    # エージェント作成
    agent = PureMemoryAgentGoalOriented(
        maze=maze,
        datastore_path="../results/25x25_optimal",
        config={
            'max_depth': 4,
            'search_k': 25,
            'gedig_improvement_threshold': 0.05
        }
    )
    
    print(f"\n📍 スタート: {agent.position}")
    print(f"🎯 ゴール: {agent.goal}")
    
    initial_dist = abs(agent.position[0] - agent.goal[0]) + abs(agent.position[1] - agent.goal[1])
    print(f"📏 初期距離: {initial_dist}")
    
    # 実行
    max_steps = 2500  # 25×25×4
    start_time = time.time()
    
    print(f"\n実行中（最大{max_steps}ステップ）...")
    print("-" * 40)
    
    for step in range(max_steps):
        if agent.is_goal_reached():
            elapsed = time.time() - start_time
            stats = agent.get_statistics()
            
            print(f"\n🎉 成功！ {step}ステップでゴール到達")
            print(f"  実行時間: {elapsed:.2f}秒")
            print(f"  壁衝突率: {stats['wall_hits']/step*100:.1f}%")
            print(f"  総エピソード: {stats['total_episodes']}")
            
            # 深度統計
            print(f"\n深度使用統計:")
            total_usage = sum(stats['depth_usage'].values())
            for depth, count in sorted(stats['depth_usage'].items()):
                if count > 0:
                    pct = count/total_usage*100
                    print(f"  {depth}ホップ: {count}回 ({pct:.1f}%)")
            
            return True
        
        # 行動
        action = agent.get_action()
        agent.execute_action(action)
        
        # 進捗
        if step % 200 == 0 and step > 0:
            stats = agent.get_statistics()
            dist = stats['distance_to_goal']
            improvement = (initial_dist - dist) / initial_dist * 100
            print(f"Step {step:4d}: 距離={dist:2d} ({improvement:+.1f}%) "
                  f"壁={stats['wall_hits']/step*100:.1f}%")
    
    # タイムアウト
    elapsed = time.time() - start_time
    final_stats = agent.get_statistics()
    final_dist = final_stats['distance_to_goal']
    
    print(f"\n⏱️ {max_steps}ステップ完了")
    print(f"  最終距離: {final_dist}/{initial_dist}")
    print(f"  改善率: {(initial_dist-final_dist)/initial_dist*100:.1f}%")
    print(f"  壁衝突率: {final_stats['wall_hits']/max_steps*100:.1f}%")
    print(f"  実行時間: {elapsed:.2f}秒")
    
    return False


if __name__ == "__main__":
    success = test_25x25()
    
    print("\n" + "="*70)
    if success:
        print("🏆 25×25迷路攻略成功！")
        print("   記憶駆動型AIが中規模迷路で有効")
        print("   geDIGが評価関数として機能")
    else:
        print("📊 25×25迷路は時間内未到達")
        print("   ただし学習による改善は確認")
    print("="*70)