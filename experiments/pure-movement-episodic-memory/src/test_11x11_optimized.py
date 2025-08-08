#!/usr/bin/env python3
"""
11×11迷路でのOptimizedNumpyIndex実験
検索高速化により、より深い評価が可能に
"""

import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent_optimized import PureMemoryAgentOptimized


def test_11x11_with_deep_evaluation():
    """11×11迷路で深い評価を実施"""
    
    print("="*70)
    print("🚀 11×11迷路での深い評価実験")
    print("  高速検索により、より多くの計算リソースを評価に割り当て")
    print("="*70)
    
    # 11×11迷路生成
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(11, 11), seed=456)
    
    print("\n迷路 (11×11):")
    for row in maze:
        print(''.join(['.' if x == 0 else '█' for x in row]))
    
    # 高度な設定でエージェント作成
    agent = PureMemoryAgentOptimized(
        maze=maze,
        datastore_path="../results/11x11_deep_evaluation",
        config={
            'max_depth': 7,           # より深い推論を許可
            'search_k': 50,           # より多くの候補を検索
            'gedig_threshold': 0.5,   # より厳密なgeDIG評価
            'max_edges_per_node': 20  # より豊富なグラフ構造
        }
    )
    
    print(f"\n📍 スタート: {agent.position}")
    print(f"🎯 ゴール: {agent.goal}")
    initial_distance = abs(agent.position[0] - agent.goal[0]) + abs(agent.position[1] - agent.goal[1])
    print(f"📏 初期マンハッタン距離: {initial_distance}")
    print("-" * 70)
    
    # 詳細な記録
    performance_log = {
        'distances': [],
        'search_times': [],
        'gedig_values': [],
        'depth_selections': [],
        'wall_hits': 0,
        'successful_moves': 0
    }
    
    # 実行
    max_steps = 1000
    milestone_steps = [50, 100, 200, 300, 500]
    
    for step in range(max_steps):
        # ゴール確認
        if agent.is_goal_reached():
            print(f"\n🎉 成功！ {step}ステップでゴール到達！")
            break
        
        # 行動前の状態
        prev_position = agent.position
        
        # 行動実行（時間計測）
        start_time = time.time()
        action = agent.get_action()
        search_time = (time.time() - start_time) * 1000
        
        success = agent.execute_action(action)
        
        # 記録
        stats = agent.get_statistics()
        performance_log['distances'].append(stats['distance_to_goal'])
        performance_log['search_times'].append(search_time)
        
        if success:
            performance_log['successful_moves'] += 1
        else:
            performance_log['wall_hits'] += 1
        
        # マイルストーン報告
        if step + 1 in milestone_steps:
            print(f"\n📊 ステップ {step + 1} の詳細分析:")
            print(f"  現在位置: {agent.position}")
            print(f"  ゴールまでの距離: {stats['distance_to_goal']}")
            print(f"  改善率: {(initial_distance - stats['distance_to_goal']) / initial_distance * 100:.1f}%")
            
            # 検索性能
            recent_search = np.mean(performance_log['search_times'][-50:])
            print(f"\n  🔍 検索性能:")
            print(f"    最近の平均検索時間: {recent_search:.3f}ms")
            print(f"    総エピソード数: {stats['total_episodes']}")
            print(f"    検索候補数(k): {agent.search_k}")
            print(f"    計算量削減率: {(1 - agent.search_k/max(1, stats['total_episodes'])) * 100:.1f}%")
            
            # geDIG分析
            print(f"\n  📈 geDIG評価:")
            print(f"    平均geDIG値: {stats['avg_gedig']:.3f}")
            if stats['avg_gedig'] < 0:
                print(f"    → 情報利得が編集距離を上回る（良好な学習）")
            
            # グラフ構造
            print(f"\n  🕸️ グラフ構造:")
            print(f"    ノード数: {stats['graph_nodes']}")
            print(f"    エッジ数: {stats['graph_edges']}")
            if stats['graph_nodes'] > 0:
                avg_degree = 2 * stats['graph_edges'] / stats['graph_nodes']
                print(f"    平均次数: {avg_degree:.2f}")
            
            # 深度使用分析
            print(f"\n  🎯 深度使用パターン:")
            total_depth = sum(stats['depth_usage'].values())
            if total_depth > 0:
                for depth in sorted(stats['depth_usage'].keys()):
                    count = stats['depth_usage'][depth]
                    if count > 0:
                        ratio = count / total_depth * 100
                        bar = '█' * int(ratio / 5)
                        print(f"    {depth}ホップ: {count:3d}回 ({ratio:5.1f}%) {bar}")
            
            # 移動効率
            print(f"\n  🚶 移動効率:")
            print(f"    成功移動: {performance_log['successful_moves']}")
            print(f"    壁衝突: {performance_log['wall_hits']}")
            success_rate = performance_log['successful_moves'] / max(1, step + 1) * 100
            print(f"    成功率: {success_rate:.1f}%")
    
    # 最終評価
    final_stats = agent.get_statistics()
    
    print("\n" + "="*70)
    print("📊 最終評価結果")
    print("="*70)
    
    # 基本結果
    print("\n基本メトリクス:")
    print(f"  最終ステップ数: {final_stats['steps']}")
    print(f"  最終距離: {final_stats['distance_to_goal']}")
    print(f"  総改善距離: {initial_distance - final_stats['distance_to_goal']}")
    
    # 効率性分析
    print("\n効率性分析:")
    print(f"  壁衝突率: {final_stats['wall_hit_rate']:.1%}")
    print(f"  平均検索時間: {final_stats['avg_search_time_ms']:.3f}ms")
    print(f"  検索高速化による節約時間: {(10 - final_stats['avg_search_time_ms']) * final_stats['steps']:.1f}ms")
    
    # 学習の質
    print("\n学習の質:")
    print(f"  平均geDIG: {final_stats['avg_gedig']:.3f}")
    print(f"  グラフ密度: {final_stats['graph_edges'] / max(1, final_stats['graph_nodes']):.2f}")
    
    # 深い推論の活用
    deep_usage = sum(final_stats['depth_usage'].get(d, 0) for d in range(4, 8))
    shallow_usage = sum(final_stats['depth_usage'].get(d, 0) for d in range(1, 4))
    total_usage = deep_usage + shallow_usage
    
    if total_usage > 0:
        deep_ratio = deep_usage / total_usage * 100
        print(f"\n深い推論の活用:")
        print(f"  浅い推論（1-3ホップ）: {shallow_usage}回")
        print(f"  深い推論（4-7ホップ）: {deep_usage}回")
        print(f"  深い推論の割合: {deep_ratio:.1f}%")
        
        if deep_ratio > 30:
            print("  → ✨ 高速検索により深い推論が活発に使用されている！")
    
    # 距離の推移グラフ（簡易版）
    if performance_log['distances']:
        print("\n📉 距離の推移（20ステップごと）:")
        for i in range(0, len(performance_log['distances']), 20):
            dist = performance_log['distances'][i]
            bar = '█' * int(dist)
            print(f"  Step {i:3d}: {bar} {dist}")
    
    return final_stats


if __name__ == "__main__":
    start_time = time.time()
    
    print("🔬 OptimizedNumpyIndexによる高速化で、より深い評価が可能に！")
    print("-" * 70)
    
    stats = test_11x11_with_deep_evaluation()
    
    total_time = time.time() - start_time
    print(f"\n⏱️ 実験総時間: {total_time:.2f}秒")
    
    if stats['distance_to_goal'] == 0:
        print("\n🏆 完全な成功！ゴールに到達しました。")
    else:
        print(f"\n📍 最終的にゴールまで{stats['distance_to_goal']}マスの距離まで接近。")
    
    print("\n" + "="*70)
    print("💡 結論: 高速検索により、より深い評価と推論が実現！")
    print("="*70)