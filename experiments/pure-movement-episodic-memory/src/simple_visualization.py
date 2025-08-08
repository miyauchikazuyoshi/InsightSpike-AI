#!/usr/bin/env python3
"""
シンプルなASCIIビジュアライゼーション
"""

import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent_optimized import PureMemoryAgentOptimized


def visualize_ascii():
    """ASCII形式でのビジュアライゼーション"""
    
    print("="*70)
    print("📊 実験結果のASCIIビジュアライゼーション")
    print("="*70)
    
    # 11×11迷路生成
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(11, 11), seed=789)
    
    # エージェント作成
    agent = PureMemoryAgentOptimized(
        maze=maze,
        datastore_path="../results/ascii_viz",
        config={
            'max_depth': 5,
            'search_k': 30,
            'gedig_threshold': 0.5
        }
    )
    
    initial_distance = abs(agent.position[0] - agent.goal[0]) + \
                      abs(agent.position[1] - agent.goal[1])
    
    # 初期迷路表示
    print("\n🗺️ 初期迷路:")
    display_maze_with_position(maze, agent.position, agent.goal, [])
    print(f"\n📍 スタート: {agent.position}")
    print(f"🎯 ゴール: {agent.goal}")
    print(f"📏 初期距離: {initial_distance}")
    
    # データ収集
    distances = []
    search_times = []
    positions = [agent.position]
    depth_counts = {i: 0 for i in range(1, 6)}
    
    print("\n" + "="*70)
    print("🚀 実行開始")
    print("="*70)
    
    # 100ステップ実行
    for step in range(100):
        if agent.is_goal_reached():
            print(f"\n🎉 成功！ {step}ステップでゴール到達！")
            break
        
        # 行動実行
        start = time.time()
        action = agent.get_action()
        search_time = (time.time() - start) * 1000
        agent.execute_action(action)
        
        # データ記録
        stats = agent.get_statistics()
        distances.append(stats['distance_to_goal'])
        search_times.append(search_time)
        positions.append(agent.position)
        
        # 深度使用記録
        for depth, count in stats['depth_usage'].items():
            depth_counts[depth] = count
        
        # 20ステップごとに表示
        if (step + 1) % 20 == 0:
            print(f"\n--- ステップ {step + 1} ---")
            display_maze_with_position(maze, agent.position, agent.goal, positions)
            
            print(f"\n📊 現在の統計:")
            print(f"  距離: {stats['distance_to_goal']} (改善: {initial_distance - stats['distance_to_goal']})")
            print(f"  検索時間: {np.mean(search_times[-20:]):.2f}ms")
            print(f"  geDIG: {stats['avg_gedig']:.3f}")
            
            # 距離グラフ（簡易版）
            print(f"\n📉 距離の推移:")
            display_distance_graph(distances[-20:])
    
    # 最終結果
    final_stats = agent.get_statistics()
    
    print("\n" + "="*70)
    print("📊 最終結果")
    print("="*70)
    
    # 最終迷路
    print("\n🗺️ 最終状態:")
    display_maze_with_position(maze, agent.position, agent.goal, positions)
    
    # 統計サマリー
    print(f"\n📈 性能統計:")
    print(f"  ゴール到達: {'✅ 成功' if agent.is_goal_reached() else f'❌ 距離 {final_stats["distance_to_goal"]}'}")
    print(f"  総ステップ: {final_stats['steps']}")
    print(f"  壁衝突率: {final_stats['wall_hit_rate']:.1%}")
    
    print(f"\n🔍 検索性能:")
    print(f"  平均時間: {np.mean(search_times):.2f}ms")
    print(f"  最小時間: {np.min(search_times):.2f}ms")
    print(f"  最大時間: {np.max(search_times):.2f}ms")
    
    print(f"\n📊 学習品質:")
    print(f"  平均geDIG: {final_stats['avg_gedig']:.3f}")
    if final_stats['avg_gedig'] < 0:
        print(f"  → ✨ 情報利得が編集距離を上回る（良好）")
    
    # 深度使用グラフ
    print(f"\n🎯 深度使用パターン:")
    total_depth = sum(depth_counts.values())
    if total_depth > 0:
        for depth in sorted(depth_counts.keys()):
            count = depth_counts[depth]
            if count > 0:
                ratio = count / total_depth * 100
                bar = '█' * int(ratio / 2)
                print(f"  {depth}ホップ: {bar} {ratio:.1f}% ({count}回)")
    
    # 計算量削減
    k = agent.search_k
    n = final_stats['total_episodes']
    if n > 0:
        reduction = (1 - k/n) * 100
        print(f"\n⚡ 高速化効果:")
        print(f"  O(n) → O(k): n={n}, k={k}")
        print(f"  計算量削減: {reduction:.1f}%")
        print(f"  推定高速化: {n/k:.1f}倍")
    
    # 距離推移の全体像
    if distances:
        print(f"\n📉 距離推移の全体像:")
        display_full_distance_graph(distances)


def display_maze_with_position(maze, position, goal, path):
    """迷路と現在位置を表示"""
    height, width = maze.shape
    
    # パスを辞書に変換
    path_dict = {}
    for i, pos in enumerate(path):
        path_dict[pos] = i
    
    for i in range(height):
        row_str = ""
        for j in range(width):
            if (i, j) == position:
                row_str += "◎"  # 現在位置
            elif (i, j) == goal:
                row_str += "★"  # ゴール
            elif (i, j) in path_dict:
                # パスの古さを表現
                age = len(path) - path_dict[(i, j)]
                if age < 10:
                    row_str += "◦"
                else:
                    row_str += "·"
            elif maze[i, j] == 1:
                row_str += "█"  # 壁
            else:
                row_str += " "  # 通路
        print(row_str)


def display_distance_graph(distances):
    """距離のASCIIグラフ"""
    if not distances:
        return
    
    max_dist = max(distances) if distances else 1
    min_dist = min(distances) if distances else 0
    
    # 5段階で表示
    levels = 5
    for level in range(levels, 0, -1):
        threshold = min_dist + (max_dist - min_dist) * level / levels
        line = ""
        for d in distances:
            if d >= threshold:
                line += "█"
            else:
                line += " "
        print(f"  {int(threshold):2d} |{line}|")
    print(f"     " + "─" * len(distances))


def display_full_distance_graph(distances):
    """全体の距離推移グラフ"""
    if not distances:
        return
    
    # 10ステップごとに集約
    aggregated = []
    for i in range(0, len(distances), 10):
        chunk = distances[i:i+10]
        if chunk:
            aggregated.append(np.mean(chunk))
    
    if not aggregated:
        return
    
    max_dist = max(aggregated)
    
    for val in aggregated:
        bar_length = int(val / max_dist * 40) if max_dist > 0 else 0
        bar = '█' * bar_length
        print(f"  {val:5.1f} |{bar}")


if __name__ == "__main__":
    visualize_ascii()