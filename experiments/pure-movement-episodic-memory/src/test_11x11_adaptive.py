#!/usr/bin/env python3
"""
11×11迷路での適応的geDIG深度選択テスト
"""

import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
import sys
import os

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent import PureMemoryAgent
from pure_memory_agent_adaptive import PureMemoryAgentAdaptive


def test_11x11_comparison():
    """11×11迷路で固定深度vs適応的深度を比較"""
    
    print("="*70)
    print("🎯 11×11迷路 適応的geDIG深度選択 実験")
    print("="*70)
    
    # 結果保存ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = Path(f"../results/11x11_adaptive_{timestamp}")
    base_path.mkdir(parents=True, exist_ok=True)
    
    # 11×11迷路生成
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(11, 11), seed=42)
    
    print("\n🗺️ 迷路 (11×11):")
    for row in maze:
        print(' '.join(['.' if x == 0 else '█' for x in row]))
    
    # 最大ステップ数
    max_steps = 1000
    
    print(f"\n⚙️ 設定:")
    print(f"  最大ステップ数: {max_steps}")
    print(f"  検索数(k): 20")
    
    # 実験結果格納
    results = {}
    
    # ============================================================
    # 1. 固定深度エージェント（3ホップ）
    # ============================================================
    print("\n" + "="*70)
    print("📌 テスト1: 固定深度エージェント（3ホップ固定）")
    print("="*70)
    
    agent_fixed = PureMemoryAgent(
        maze=maze.copy(),
        datastore_path=str(base_path / "fixed_depth"),
        config={
            'max_depth': 3,
            'search_k': 20
        }
    )
    
    print(f"スタート: {agent_fixed.position}")
    print(f"ゴール: {agent_fixed.goal}")
    
    start_time = time.time()
    
    for step in range(max_steps):
        if agent_fixed.is_goal_reached():
            success_fixed = True
            break
        
        action = agent_fixed.get_action()
        agent_fixed.execute_action(action)
        
        # 進捗報告
        if step % 100 == 0 and step > 0:
            stats = agent_fixed.get_statistics()
            print(f"  ステップ {step}: 距離={stats['distance_to_goal']}, "
                  f"壁衝突={stats['wall_hits']}回 "
                  f"({stats['wall_hits']/step*100:.1f}%)")
    else:
        success_fixed = False
    
    time_fixed = time.time() - start_time
    stats_fixed = agent_fixed.get_statistics()
    
    results['fixed'] = {
        'success': success_fixed,
        'steps': step if success_fixed else max_steps,
        'time': time_fixed,
        'wall_hits': stats_fixed['wall_hits'],
        'wall_hit_rate': stats_fixed['wall_hits'] / max(step, 1),
        'episodes': stats_fixed['total_episodes'],
        'path_length': stats_fixed['path_length'],
        'final_distance': stats_fixed['distance_to_goal'],
        'depth_usage': stats_fixed['depth_usage']
    }
    
    if success_fixed:
        print(f"\n✅ 成功！ {step}ステップで到達")
    else:
        print(f"\n❌ 失敗... 最終距離: {stats_fixed['distance_to_goal']}")
    
    print(f"  壁衝突率: {results['fixed']['wall_hit_rate']:.1%}")
    print(f"  エピソード数: {results['fixed']['episodes']}")
    print(f"  実行時間: {results['fixed']['time']:.2f}秒")
    
    # ============================================================
    # 2. 適応的深度エージェント（geDIG基準）
    # ============================================================
    print("\n" + "="*70)
    print("🧠 テスト2: 適応的深度エージェント（geDIG基準）")
    print("="*70)
    
    agent_adaptive = PureMemoryAgentAdaptive(
        maze=maze.copy(),
        datastore_path=str(base_path / "adaptive_depth"),
        config={
            'max_depth': 5,
            'search_k': 20,
            'gedig_improvement_threshold': 0.05  # 5%改善で採用
        }
    )
    
    print(f"スタート: {agent_adaptive.position}")
    print(f"ゴール: {agent_adaptive.goal}")
    print(f"geDIG改善閾値: 5%")
    
    start_time = time.time()
    
    # 深度選択の詳細記録
    depth_history = []
    
    for step in range(max_steps):
        if agent_adaptive.is_goal_reached():
            success_adaptive = True
            break
        
        # 深度選択を記録
        before_selections = len(agent_adaptive.stats['adaptive_depth_selections'])
        
        action = agent_adaptive.get_action()
        agent_adaptive.execute_action(action)
        
        # 新しい深度選択があれば記録
        if len(agent_adaptive.stats['adaptive_depth_selections']) > before_selections:
            selected_depth = agent_adaptive.stats['adaptive_depth_selections'][-1]
            depth_history.append((step, selected_depth))
        
        # 進捗報告
        if step % 100 == 0 and step > 0:
            stats = agent_adaptive.get_statistics()
            avg_depth = stats.get('avg_adaptive_depth', 0)
            print(f"  ステップ {step}: 距離={stats['distance_to_goal']}, "
                  f"壁衝突={stats['wall_hits']}回 "
                  f"({stats['wall_hits']/step*100:.1f}%), "
                  f"平均深度={avg_depth:.2f}")
    else:
        success_adaptive = False
    
    time_adaptive = time.time() - start_time
    stats_adaptive = agent_adaptive.get_statistics()
    
    results['adaptive'] = {
        'success': success_adaptive,
        'steps': step if success_adaptive else max_steps,
        'time': time_adaptive,
        'wall_hits': stats_adaptive['wall_hits'],
        'wall_hit_rate': stats_adaptive['wall_hits'] / max(step, 1),
        'episodes': stats_adaptive['total_episodes'],
        'path_length': stats_adaptive['path_length'],
        'final_distance': stats_adaptive['distance_to_goal'],
        'depth_usage': stats_adaptive['depth_usage'],
        'avg_adaptive_depth': stats_adaptive.get('avg_adaptive_depth', 0),
        'adaptive_selections': stats_adaptive.get('adaptive_selections', [])
    }
    
    if success_adaptive:
        print(f"\n✅ 成功！ {step}ステップで到達")
    else:
        print(f"\n❌ 失敗... 最終距離: {stats_adaptive['distance_to_goal']}")
    
    print(f"  壁衝突率: {results['adaptive']['wall_hit_rate']:.1%}")
    print(f"  エピソード数: {results['adaptive']['episodes']}")
    print(f"  実行時間: {results['adaptive']['time']:.2f}秒")
    print(f"  平均選択深度: {results['adaptive']['avg_adaptive_depth']:.2f}")
    
    # ============================================================
    # 3. 比較分析
    # ============================================================
    print("\n" + "="*70)
    print("📊 比較結果")
    print("="*70)
    
    # 成功率
    print("\n🎯 成功/失敗:")
    print(f"  固定深度:   {'✅ 成功' if results['fixed']['success'] else '❌ 失敗'}")
    print(f"  適応的深度: {'✅ 成功' if results['adaptive']['success'] else '❌ 失敗'}")
    
    # ステップ数比較
    if results['fixed']['success'] or results['adaptive']['success']:
        print("\n📏 ゴール到達ステップ数:")
        if results['fixed']['success']:
            print(f"  固定深度:   {results['fixed']['steps']}ステップ")
        if results['adaptive']['success']:
            print(f"  適応的深度: {results['adaptive']['steps']}ステップ")
            
        # 改善率計算
        if results['fixed']['success'] and results['adaptive']['success']:
            improvement = (results['fixed']['steps'] - results['adaptive']['steps']) / results['fixed']['steps'] * 100
            if improvement > 0:
                print(f"  → 適応的深度が {improvement:.1f}% 改善！")
            elif improvement < 0:
                print(f"  → 固定深度の方が {-improvement:.1f}% 良い")
            else:
                print(f"  → 同じステップ数")
    
    # 壁衝突率比較
    print("\n🧱 壁衝突率:")
    print(f"  固定深度:   {results['fixed']['wall_hit_rate']:.1%}")
    print(f"  適応的深度: {results['adaptive']['wall_hit_rate']:.1%}")
    wall_improvement = (results['fixed']['wall_hit_rate'] - results['adaptive']['wall_hit_rate']) / results['fixed']['wall_hit_rate'] * 100
    if wall_improvement > 0:
        print(f"  → 適応的深度が {wall_improvement:.1f}% 改善！")
    
    # 計算効率
    print("\n⏱️ 実行時間:")
    print(f"  固定深度:   {results['fixed']['time']:.2f}秒")
    print(f"  適応的深度: {results['adaptive']['time']:.2f}秒")
    
    # 深度使用分析（適応的エージェント）
    if results['adaptive']['adaptive_selections']:
        print("\n🔍 適応的深度の選択パターン:")
        selections = results['adaptive']['adaptive_selections']
        
        # 深度分布
        depth_counts = {}
        for d in selections:
            depth_counts[d] = depth_counts.get(d, 0) + 1
        
        for depth in sorted(depth_counts.keys()):
            percentage = depth_counts[depth] / len(selections) * 100
            bar = '█' * int(percentage / 2)
            print(f"  {depth}ホップ: {depth_counts[depth]:3d}回 ({percentage:5.1f}%) {bar}")
        
        # 序盤と終盤の比較
        if len(selections) > 20:
            early = selections[:10]
            late = selections[-10:]
            print(f"\n  序盤の平均深度: {np.mean(early):.2f}")
            print(f"  終盤の平均深度: {np.mean(late):.2f}")
            
            change = np.mean(late) - np.mean(early)
            if change > 0.3:
                print("  → 📈 学習とともに深い探索を活用")
            elif change < -0.3:
                print("  → 📉 学習とともに浅い探索に収束")
            else:
                print("  → 📊 安定した深度選択")
    
    # geDIG評価の分析
    if agent_adaptive.stats.get('gedig_evaluations'):
        evaluations = agent_adaptive.stats['gedig_evaluations']
        improvements = []
        
        for eval_history in evaluations[:50]:  # 最初の50個
            if len(eval_history) > 1:
                base = eval_history[0][1]
                best = min(h[1] for h in eval_history)
                improvement = (base - best) / (base + 0.001)
                improvements.append(improvement)
        
        if improvements:
            print(f"\n💡 geDIG改善率:")
            print(f"  平均: {np.mean(improvements):.3f}")
            print(f"  最大: {max(improvements):.3f}")
            print(f"  最小: {min(improvements):.3f}")
    
    # 結果をJSON保存
    with open(base_path / "comparison_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 結果を保存: {base_path}")
    
    return results


if __name__ == "__main__":
    results = test_11x11_comparison()
    
    print("\n" + "="*70)
    print("🏁 実験完了！")
    print("="*70)
    
    # 最終評価
    if results['adaptive']['success'] and not results['fixed']['success']:
        print("⭐ 適応的深度選択が優れた性能を発揮！")
        print("   固定深度では解けなかった迷路を解決")
    elif results['adaptive']['success'] and results['fixed']['success']:
        if results['adaptive']['steps'] < results['fixed']['steps']:
            print("✨ 適応的深度選択がより効率的！")
            print("   少ないステップでゴールに到達")
        elif results['adaptive']['wall_hit_rate'] < results['fixed']['wall_hit_rate']:
            print("🎯 適応的深度選択がより正確！")
            print("   壁衝突が少ない")
        else:
            print("📊 両手法とも成功、性能は同等")
    elif not results['adaptive']['success'] and not results['fixed']['success']:
        print("🔧 両手法とも失敗...")
        print("   より長い学習時間が必要かも")
    else:
        print("📈 固定深度の方が良い結果")
        print("   パラメータ調整が必要かも")