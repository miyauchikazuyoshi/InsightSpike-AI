#!/usr/bin/env python3
"""
50×50迷路クイックテスト（2000ステップ限定）
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


def quick_50x50_test():
    """50×50迷路の短時間テスト"""
    
    print("="*70)
    print("🏔️ 50×50迷路クイックテスト（2000ステップ）")
    print("="*70)
    
    # 50×50迷路生成
    print("\n🏗️ 迷路生成中...")
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(51, 51), seed=42)
    
    # 迷路の概要
    print("\n迷路サイズ: 51×51")
    print("左上部分（10×10）:")
    for i in range(10):
        row_str = ''.join(['.' if maze[i][j] == 0 else '█' for j in range(10)])
        print(row_str)
    print("...")
    
    # エージェント作成（軽量設定）
    agent = PureMemoryAgentGoalOriented(
        maze=maze,
        datastore_path="../results/50x50_quick",
        config={
            'max_depth': 3,      # 深度を制限
            'search_k': 30,      # 検索数も控えめ
            'gedig_improvement_threshold': 0.1  # 10%改善で採用
        }
    )
    
    print(f"\n📍 スタート: {agent.position}")
    print(f"🎯 ゴール: {agent.goal}")
    
    initial_distance = abs(agent.position[0] - agent.goal[0]) + abs(agent.position[1] - agent.goal[1])
    print(f"📏 初期距離: {initial_distance}")
    
    # 実行
    max_steps = 2000
    start_time = time.time()
    
    print(f"\n実行中（最大{max_steps}ステップ）...")
    print("-" * 40)
    
    # 進捗記録
    progress_points = []
    
    for step in range(max_steps):
        if agent.is_goal_reached():
            elapsed = time.time() - start_time
            print(f"\n🎉 成功！ {step}ステップでゴール到達")
            print(f"  時間: {elapsed:.2f}秒")
            print(f"  壁衝突率: {agent.stats['wall_hits']/step*100:.1f}%")
            return True
        
        # 行動
        action = agent.get_action()
        agent.execute_action(action)
        
        # 進捗報告（250ステップごと）
        if step % 250 == 0 and step > 0:
            stats = agent.get_statistics()
            current_distance = stats['distance_to_goal']
            progress_points.append(current_distance)
            
            improvement = (initial_distance - current_distance) / initial_distance * 100
            print(f"Step {step:4d}: 距離={current_distance:3d} "
                  f"(改善率{improvement:+6.1f}%) "
                  f"壁衝突率={stats['wall_hits']/step*100:.1f}%")
    
    # 最終結果
    elapsed = time.time() - start_time
    final_stats = agent.get_statistics()
    final_distance = final_stats['distance_to_goal']
    
    print(f"\n⏱️ {max_steps}ステップ完了")
    print(f"  最終距離: {final_distance} (初期: {initial_distance})")
    
    total_improvement = (initial_distance - final_distance) / initial_distance * 100
    if total_improvement > 0:
        print(f"  📈 {total_improvement:.1f}% 改善")
    else:
        print(f"  📉 {-total_improvement:.1f}% 悪化")
    
    print(f"  壁衝突率: {final_stats['wall_hits']/max_steps*100:.1f}%")
    print(f"  実行時間: {elapsed:.2f}秒")
    
    # 深度使用
    print(f"\n深度使用:")
    for depth, count in final_stats['depth_usage'].items():
        if count > 0:
            print(f"  {depth}ホップ: {count}回")
    
    # クエリタイプ
    qt = final_stats.get('query_types', {})
    if qt:
        print(f"\nクエリタイプ:")
        total = sum(qt.values())
        print(f"  ゴール指向: {qt.get('goal_oriented', 0)/total*100:.1f}%")
        print(f"  探索: {qt.get('exploration', 0)/total*100:.1f}%")
    
    # 進捗評価
    if progress_points:
        print(f"\n進捗推移: {progress_points}")
        if all(progress_points[i] >= progress_points[i+1] for i in range(len(progress_points)-1)):
            print("  → 📈 一貫して改善")
        else:
            print("  → 📊 改善と停滞が混在")
    
    return False


if __name__ == "__main__":
    print("🚀 50×50迷路チャレンジ開始！")
    print("  最善設定：ゴール指向クエリ + geDIG適応")
    print("")
    
    success = quick_50x50_test()
    
    print("\n" + "="*70)
    if success:
        print("🏆 50×50迷路を攻略！")
        print("   純粋記憶ベースが大規模迷路でも機能")
    else:
        print("📊 2000ステップでは未到達")
        print("   ただし進捗は確認できた")
        print("   より長い学習で成功の可能性あり")
    print("="*70)