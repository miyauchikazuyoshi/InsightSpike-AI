#!/usr/bin/env python3
"""
11×11迷路での軽量版実験
"""

import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent_optimized import PureMemoryAgentOptimized


def test_11x11_light():
    """11×11迷路で軽量テスト"""
    
    print("="*60)
    print("🚀 11×11迷路実験（軽量版）")
    print("="*60)
    
    # 11×11迷路
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(11, 11), seed=789)
    
    print("\n迷路 (11×11):")
    for row in maze:
        print(''.join(['.' if x == 0 else '█' for x in row]))
    
    # 軽量設定
    agent = PureMemoryAgentOptimized(
        maze=maze,
        datastore_path="../results/11x11_light",
        config={
            'max_depth': 4,
            'search_k': 20,
            'gedig_threshold': 0.6,
            'max_edges_per_node': 10
        }
    )
    
    print(f"\n📍 スタート: {agent.position}")
    print(f"🎯 ゴール: {agent.goal}")
    initial_distance = abs(agent.position[0] - agent.goal[0]) + abs(agent.position[1] - agent.goal[1])
    print(f"📏 初期距離: {initial_distance}")
    print("-" * 40)
    
    # 200ステップ実行
    search_times = []
    distances = []
    
    for step in range(200):
        if agent.is_goal_reached():
            print(f"\n✅ {step}ステップでゴール到達！")
            break
        
        # 行動
        start = time.time()
        action = agent.get_action()
        search_time = (time.time() - start) * 1000
        search_times.append(search_time)
        
        agent.execute_action(action)
        
        # 距離記録
        stats = agent.get_statistics()
        distances.append(stats['distance_to_goal'])
        
        # 50ステップごとに報告
        if step % 50 == 49:
            avg_search = np.mean(search_times[-50:])
            print(f"\nStep {step+1}:")
            print(f"  距離: {stats['distance_to_goal']} (改善: {initial_distance - stats['distance_to_goal']})")
            print(f"  検索時間: {avg_search:.3f}ms")
            print(f"  geDIG: {stats['avg_gedig']:.3f}")
            print(f"  グラフ: {stats['graph_nodes']}ノード, {stats['graph_edges']}エッジ")
            
            # 深度使用
            total = sum(stats['depth_usage'].values())
            if total > 0:
                deep = sum(stats['depth_usage'].get(d, 0) for d in range(3, 5))
                print(f"  深い推論: {deep/total*100:.1f}%")
    
    # 最終結果
    final_stats = agent.get_statistics()
    
    print("\n" + "="*60)
    print("📊 最終結果")
    print("="*60)
    
    print(f"  最終距離: {final_stats['distance_to_goal']}")
    print(f"  改善距離: {initial_distance - final_stats['distance_to_goal']}")
    print(f"  壁衝突率: {final_stats['wall_hit_rate']:.1%}")
    print(f"  平均検索: {np.mean(search_times):.3f}ms")
    
    # 効率性
    print(f"\n⚡ 高速検索の効果:")
    print(f"  k={agent.search_k}, n={final_stats['total_episodes']}")
    print(f"  計算量削減: {(1 - agent.search_k/max(1, final_stats['total_episodes'])) * 100:.1f}%")
    
    # 学習の質
    if final_stats['avg_gedig'] < 0:
        print(f"\n✨ 良好な学習: geDIG={final_stats['avg_gedig']:.3f} < 0")
        print("  情報利得が編集距離を上回っている")


if __name__ == "__main__":
    test_11x11_light()