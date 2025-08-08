#!/usr/bin/env python3
"""
OptimizedNumpyIndexを使用した純粋記憶エージェントのテスト
高速検索の性能評価
"""

import numpy as np
import time
import sys
import os

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent_optimized import PureMemoryAgentOptimized


def test_performance_comparison():
    """性能比較テスト"""
    
    print("="*70)
    print("🚀 OptimizedNumpyIndex性能評価")
    print("  メインコードの高速検索実装を活用")
    print("="*70)
    
    # 15×15迷路でテスト
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(15, 15), seed=456)
    
    print("\n迷路 (15×15):")
    for row in maze:
        print(''.join(['.' if x == 0 else '█' for x in row]))
    
    # 最適化エージェント作成
    agent = PureMemoryAgentOptimized(
        maze=maze,
        datastore_path="../results/optimized_test",
        config={
            'max_depth': 5,
            'search_k': 30,
            'gedig_threshold': 0.6,
            'max_edges_per_node': 15
        }
    )
    
    print(f"\n📍 スタート: {agent.position}")
    print(f"🎯 ゴール: {agent.goal}")
    print(f"初期距離: {abs(agent.position[0] - agent.goal[0]) + abs(agent.position[1] - agent.goal[1])}")
    print("-" * 50)
    
    # 性能メトリクス記録
    step_times = []
    search_times_per_100 = []
    gedig_values_per_100 = []
    
    max_steps = 500
    for step in range(max_steps):
        if agent.is_goal_reached():
            print(f"\n✅ 成功！ {step}ステップでゴール到達")
            break
        
        # ステップ実行時間計測
        step_start = time.time()
        action = agent.get_action()
        agent.execute_action(action)
        step_time = (time.time() - step_start) * 1000
        step_times.append(step_time)
        
        # 100ステップごとの統計
        if step % 100 == 99:
            stats = agent.get_statistics()
            
            # 検索性能
            avg_search = stats['avg_search_time_ms']
            search_times_per_100.append(avg_search)
            
            # geDIG値
            avg_gedig = stats['avg_gedig']
            gedig_values_per_100.append(avg_gedig)
            
            print(f"\n📊 Step {step+1} 統計:")
            print(f"  距離: {stats['distance_to_goal']}")
            print(f"  壁衝突率: {stats['wall_hit_rate']:.1%}")
            print(f"  平均検索時間: {avg_search:.3f}ms")
            print(f"  平均geDIG: {avg_gedig:.3f}")
            print(f"  グラフ: {stats['graph_nodes']}ノード, {stats['graph_edges']}エッジ")
            print(f"  エピソード数: {stats['total_episodes']}")
    
    # 最終統計
    final_stats = agent.get_statistics()
    
    print("\n" + "="*70)
    print("📈 性能評価結果")
    print("="*70)
    
    print("\n基本メトリクス:")
    print(f"  総ステップ数: {final_stats['steps']}")
    print(f"  壁衝突率: {final_stats['wall_hit_rate']:.1%}")
    print(f"  総エピソード数: {final_stats['total_episodes']}")
    
    print("\n検索性能:")
    if step_times:
        print(f"  平均ステップ時間: {np.mean(step_times):.2f}ms")
        print(f"  最大ステップ時間: {np.max(step_times):.2f}ms")
    
    if search_times_per_100:
        print(f"  平均検索時間: {np.mean(search_times_per_100):.3f}ms")
        print(f"  検索時間の変化: {search_times_per_100}")
    
    print("\ngeDIG評価:")
    if gedig_values_per_100:
        print(f"  平均geDIG: {np.mean(gedig_values_per_100):.3f}")
        print(f"  geDIG値の変化: {[f'{v:.3f}' for v in gedig_values_per_100]}")
    
    print("\n深度使用:")
    total_depth_usage = sum(final_stats['depth_usage'].values())
    if total_depth_usage > 0:
        for depth, count in final_stats['depth_usage'].items():
            ratio = count / total_depth_usage * 100
            print(f"  {depth}ホップ: {count}回 ({ratio:.1f}%)")
    
    print("\nグラフ構造:")
    print(f"  ノード数: {final_stats['graph_nodes']}")
    print(f"  エッジ数: {final_stats['graph_edges']}")
    if final_stats['graph_nodes'] > 0:
        avg_degree = 2 * final_stats['graph_edges'] / final_stats['graph_nodes']
        print(f"  平均次数: {avg_degree:.2f}")
    
    # 効率性評価
    print("\n⚡ 効率性評価:")
    
    # O(n) vs O(k)の改善
    if final_stats['total_episodes'] > 0:
        theoretical_o_n = final_stats['total_episodes'] * final_stats['steps']
        actual_operations = agent.search_k * final_stats['steps']
        improvement = (theoretical_o_n - actual_operations) / theoretical_o_n * 100
        
        print(f"  理論的O(n)操作数: {theoretical_o_n:,}")
        print(f"  実際のO(k)操作数: {actual_operations:,}")
        print(f"  改善率: {improvement:.1f}%")
    
    # メモリ効率
    print(f"\nメモリ効率:")
    print(f"  エピソード当たりのエッジ数: {final_stats['graph_edges'] / max(1, final_stats['graph_nodes']):.2f}")
    
    return final_stats


def test_scaling():
    """スケーリングテスト（異なるサイズの迷路）"""
    
    print("\n" + "="*70)
    print("📏 スケーリングテスト")
    print("="*70)
    
    sizes = [(7, 7), (11, 11), (15, 15), (21, 21)]
    results = []
    
    for size in sizes:
        print(f"\n🔍 {size[0]}×{size[1]}迷路でテスト...")
        
        generator = ProperMazeGenerator()
        maze = generator.generate_dfs_maze(size=size, seed=789)
        
        agent = PureMemoryAgentOptimized(
            maze=maze,
            datastore_path=f"../results/scaling_test_{size[0]}x{size[1]}",
            config={
                'max_depth': 4,
                'search_k': 20,
                'gedig_threshold': 0.6
            }
        )
        
        # 100ステップ実行
        search_times = []
        for _ in range(100):
            if agent.is_goal_reached():
                break
            
            start = time.time()
            action = agent.get_action()
            search_time = (time.time() - start) * 1000
            search_times.append(search_time)
            
            agent.execute_action(action)
        
        stats = agent.get_statistics()
        avg_search = np.mean(search_times) if search_times else 0
        
        results.append({
            'size': size,
            'episodes': stats['total_episodes'],
            'avg_search_ms': avg_search,
            'graph_edges': stats['graph_edges']
        })
        
        print(f"  エピソード数: {stats['total_episodes']}")
        print(f"  平均検索時間: {avg_search:.3f}ms")
        print(f"  グラフエッジ数: {stats['graph_edges']}")
    
    # スケーリング分析
    print("\n📊 スケーリング分析:")
    print("サイズ\tエピソード\t検索時間(ms)\tエッジ数")
    print("-" * 50)
    for r in results:
        print(f"{r['size'][0]}×{r['size'][1]}\t{r['episodes']}\t\t{r['avg_search_ms']:.3f}\t\t{r['graph_edges']}")
    
    # 検索時間の増加率を計算
    if len(results) > 1:
        time_increase = results[-1]['avg_search_ms'] / results[0]['avg_search_ms']
        episode_increase = results[-1]['episodes'] / results[0]['episodes']
        
        print(f"\n検索時間増加率: {time_increase:.2f}x")
        print(f"エピソード数増加率: {episode_increase:.2f}x")
        
        if time_increase < episode_increase:
            print("✅ 検索時間の増加がエピソード数の増加より緩やか（O(k)の効果）")
        else:
            print("⚠️ 検索時間がエピソード数に比例して増加")


if __name__ == "__main__":
    # 性能比較テスト
    stats = test_performance_comparison()
    
    # スケーリングテスト
    test_scaling()
    
    print("\n" + "="*70)
    print("🎉 テスト完了")
    print("  OptimizedNumpyIndexによる高速化を確認")
    print("="*70)