#!/usr/bin/env python3
"""
ゴール指向クエリのテスト
訪問回数=0、ゴール=1.0のクエリ設定
"""

import numpy as np
import time
from datetime import datetime
import sys
import os

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent_adaptive import PureMemoryAgentAdaptive
from pure_memory_agent_goal_oriented import PureMemoryAgentGoalOriented


def compare_query_strategies():
    """クエリ戦略の比較：中立 vs ゴール指向"""
    
    print("="*70)
    print("🎯 クエリ戦略比較実験")
    print("  1. 中立クエリ（訪問回数=現在値、ゴール=0.5）")
    print("  2. ゴール指向（訪問回数=0、ゴール=1.0）")
    print("="*70)
    
    # 11×11迷路で比較
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(11, 11), seed=42)
    
    print("\n🗺️ 迷路 (11×11):")
    for row in maze:
        print(' '.join(['.' if x == 0 else '█' for x in row]))
    
    max_steps = 300
    results = {}
    
    # ============================================================
    # 1. 中立クエリ（従来版）
    # ============================================================
    print("\n" + "-"*70)
    print("📌 テスト1: 中立クエリ（チートなし版）")
    print("-"*70)
    
    agent_neutral = PureMemoryAgentAdaptive(
        maze=maze.copy(),
        datastore_path="../results/neutral_query_test",
        config={
            'max_depth': 4,
            'search_k': 15,
            'gedig_improvement_threshold': 0.05
        }
    )
    
    print(f"スタート: {agent_neutral.position}, ゴール: {agent_neutral.goal}")
    
    # 実行
    for step in range(max_steps):
        if agent_neutral.is_goal_reached():
            success_neutral = True
            break
        
        action = agent_neutral.get_action()
        agent_neutral.execute_action(action)
        
        if step % 50 == 0 and step > 0:
            stats = agent_neutral.get_statistics()
            print(f"  ステップ {step}: 距離={stats['distance_to_goal']}, "
                  f"壁衝突率={stats['wall_hits']/step*100:.1f}%")
    else:
        success_neutral = False
    
    stats_neutral = agent_neutral.get_statistics()
    results['neutral'] = {
        'success': success_neutral,
        'steps': step if success_neutral else max_steps,
        'wall_hit_rate': stats_neutral['wall_hits'] / max(step, 1),
        'final_distance': stats_neutral['distance_to_goal'],
        'avg_depth': stats_neutral.get('avg_adaptive_depth', 0)
    }
    
    if success_neutral:
        print(f"✅ 成功！ {step}ステップ")
    else:
        print(f"❌ 失敗（最終距離: {stats_neutral['distance_to_goal']}）")
    print(f"  壁衝突率: {results['neutral']['wall_hit_rate']:.1%}")
    
    # ============================================================
    # 2. ゴール指向クエリ（新版）
    # ============================================================
    print("\n" + "-"*70)
    print("🎯 テスト2: ゴール指向クエリ（訪問=0、ゴール=1.0）")
    print("-"*70)
    
    agent_goal = PureMemoryAgentGoalOriented(
        maze=maze.copy(),
        datastore_path="../results/goal_oriented_test",
        config={
            'max_depth': 4,
            'search_k': 15,
            'gedig_improvement_threshold': 0.05
        }
    )
    
    print(f"スタート: {agent_goal.position}, ゴール: {agent_goal.goal}")
    
    # 実行
    for step in range(max_steps):
        if agent_goal.is_goal_reached():
            success_goal = True
            break
        
        action = agent_goal.get_action()
        agent_goal.execute_action(action)
        
        if step % 50 == 0 and step > 0:
            stats = agent_goal.get_statistics()
            print(f"  ステップ {step}: 距離={stats['distance_to_goal']}, "
                  f"壁衝突率={stats['wall_hits']/step*100:.1f}%")
    else:
        success_goal = False
    
    stats_goal = agent_goal.get_statistics()
    results['goal_oriented'] = {
        'success': success_goal,
        'steps': step if success_goal else max_steps,
        'wall_hit_rate': stats_goal['wall_hits'] / max(step, 1),
        'final_distance': stats_goal['distance_to_goal'],
        'avg_depth': stats_goal.get('avg_adaptive_depth', 0),
        'query_types': stats_goal.get('query_types', {})
    }
    
    if success_goal:
        print(f"✅ 成功！ {step}ステップ")
    else:
        print(f"❌ 失敗（最終距離: {stats_goal['distance_to_goal']}）")
    print(f"  壁衝突率: {results['goal_oriented']['wall_hit_rate']:.1%}")
    
    # クエリタイプの使用状況
    if 'query_types' in results['goal_oriented']:
        qt = results['goal_oriented']['query_types']
        total_queries = sum(qt.values())
        if total_queries > 0:
            print(f"  クエリタイプ:")
            print(f"    ゴール指向: {qt.get('goal_oriented', 0)} "
                  f"({qt.get('goal_oriented', 0)/total_queries*100:.1f}%)")
            print(f"    探索: {qt.get('exploration', 0)} "
                  f"({qt.get('exploration', 0)/total_queries*100:.1f}%)")
    
    # ============================================================
    # 3. 比較分析
    # ============================================================
    print("\n" + "="*70)
    print("📊 比較結果")
    print("="*70)
    
    # 成功率
    print("\n🎯 成功/失敗:")
    print(f"  中立クエリ:     {'✅' if results['neutral']['success'] else '❌'} "
          f"（距離: {results['neutral']['final_distance']}）")
    print(f"  ゴール指向:     {'✅' if results['goal_oriented']['success'] else '❌'} "
          f"（距離: {results['goal_oriented']['final_distance']}）")
    
    # ステップ数比較
    if results['neutral']['success'] or results['goal_oriented']['success']:
        print("\n📏 ゴール到達ステップ数:")
        if results['neutral']['success']:
            print(f"  中立クエリ: {results['neutral']['steps']}ステップ")
        if results['goal_oriented']['success']:
            print(f"  ゴール指向: {results['goal_oriented']['steps']}ステップ")
        
        if results['neutral']['success'] and results['goal_oriented']['success']:
            improvement = (results['neutral']['steps'] - results['goal_oriented']['steps']) / results['neutral']['steps'] * 100
            if improvement > 0:
                print(f"  → ゴール指向が {improvement:.1f}% 改善！")
            elif improvement < 0:
                print(f"  → 中立クエリの方が {-improvement:.1f}% 良い")
    
    # 壁衝突率
    print("\n🧱 壁衝突率:")
    print(f"  中立クエリ: {results['neutral']['wall_hit_rate']:.1%}")
    print(f"  ゴール指向: {results['goal_oriented']['wall_hit_rate']:.1%}")
    
    wall_improvement = (results['neutral']['wall_hit_rate'] - results['goal_oriented']['wall_hit_rate']) / results['neutral']['wall_hit_rate'] * 100
    if wall_improvement > 0:
        print(f"  → ゴール指向が {wall_improvement:.1f}% 改善！")
    
    # 最終距離の改善
    dist_improvement = results['neutral']['final_distance'] - results['goal_oriented']['final_distance']
    if dist_improvement > 0:
        print(f"\n📍 最終距離: ゴール指向が {dist_improvement} マス近い！")
    elif dist_improvement < 0:
        print(f"\n📍 最終距離: 中立クエリが {-dist_improvement} マス近い")
    
    # クエリ設定の違い
    print("\n💡 クエリ設定の違い:")
    print("  中立クエリ:")
    print("    - 訪問回数: 現在の訪問状況を反映")
    print("    - ゴール: 0.5（中立、チートなし）")
    print("  ゴール指向:")
    print("    - 訪問回数: 0（未訪問エリアを探索）")
    print("    - ゴール: 1.0（ゴール関連の記憶を優先）")
    
    return results


if __name__ == "__main__":
    results = compare_query_strategies()
    
    print("\n" + "="*70)
    print("🏁 実験完了！")
    print("="*70)
    
    # 最終評価
    neutral_success = results['neutral']['success']
    goal_success = results['goal_oriented']['success']
    
    if goal_success and not neutral_success:
        print("⭐ ゴール指向クエリが優れた性能！")
        print("   未訪問エリアとゴール記憶の優先が効果的")
    elif goal_success and neutral_success:
        if results['goal_oriented']['steps'] < results['neutral']['steps']:
            print("✨ ゴール指向クエリがより効率的！")
        else:
            print("📊 両方成功、性能は同等")
    elif not goal_success and not neutral_success:
        print("🔧 両方失敗... より長い学習が必要")
        if results['goal_oriented']['final_distance'] < results['neutral']['final_distance']:
            print("   ただし、ゴール指向の方がゴールに近い")
    else:
        print("📈 中立クエリの方が良い結果")