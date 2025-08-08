#!/usr/bin/env python3
"""
OptimizedNumpyIndex簡易テスト
"""

import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent_optimized import PureMemoryAgentOptimized


def quick_test():
    """7×7迷路で簡易テスト"""
    
    print("="*60)
    print("🚀 OptimizedNumpyIndex簡易テスト")
    print("="*60)
    
    # 7×7迷路
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(7, 7), seed=123)
    
    print("\n迷路 (7×7):")
    for row in maze:
        print(''.join(['.' if x == 0 else '█' for x in row]))
    
    # エージェント作成
    agent = PureMemoryAgentOptimized(
        maze=maze,
        datastore_path="../results/optimized_quick",
        config={
            'max_depth': 3,
            'search_k': 15
        }
    )
    
    print(f"\n📍 スタート: {agent.position}")
    print(f"🎯 ゴール: {agent.goal}")
    
    # 100ステップ実行
    search_times = []
    for step in range(100):
        if agent.is_goal_reached():
            print(f"\n✅ {step}ステップでゴール到達！")
            break
        
        start = time.time()
        action = agent.get_action()
        search_time = (time.time() - start) * 1000
        search_times.append(search_time)
        
        agent.execute_action(action)
        
        if step % 20 == 19:
            stats = agent.get_statistics()
            print(f"Step {step+1}: 距離={stats['distance_to_goal']}, "
                  f"検索時間={np.mean(search_times[-20:]):.3f}ms")
    
    # 最終統計
    stats = agent.get_statistics()
    
    print("\n📊 最終統計:")
    print(f"  総ステップ: {stats['steps']}")
    print(f"  壁衝突率: {stats['wall_hit_rate']:.1%}")
    print(f"  平均検索時間: {stats['avg_search_time_ms']:.3f}ms")
    print(f"  平均geDIG: {stats['avg_gedig']:.3f}")
    print(f"  グラフ: {stats['graph_nodes']}ノード, {stats['graph_edges']}エッジ")
    
    # 効率性の確認
    print("\n⚡ 効率性:")
    print(f"  検索でO(n)→O(k)削減")
    print(f"  k={agent.search_k}, n={stats['total_episodes']}")
    print(f"  削減率: {(1 - agent.search_k/max(1, stats['total_episodes'])) * 100:.1f}%")


if __name__ == "__main__":
    quick_test()