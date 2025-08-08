#!/usr/bin/env python3
"""
geDIG適応的深度選択実験
純粋記憶エージェントの改良版をテスト
"""

import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
import sys
import os
from typing import Dict, List

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent import PureMemoryAgent
from pure_memory_agent_adaptive import PureMemoryAgentAdaptive


def run_single_experiment(agent, maze_size: int, max_steps: int, name: str) -> Dict:
    """単一実験を実行"""
    
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    for step in range(max_steps):
        if agent.is_goal_reached():
            success = True
            break
        
        action = agent.get_action()
        agent.execute_action(action)
        
        # 進捗報告
        if step % 100 == 0 and step > 0:
            stats = agent.get_statistics()
            print(f"Step {step}: dist={stats['distance_to_goal']}, "
                  f"wall_hits={stats['wall_hits']} "
                  f"({stats['wall_hits']/step*100:.1f}%)")
    else:
        success = False
    
    # 結果収集
    total_time = time.time() - start_time
    final_stats = agent.get_statistics()
    
    result = {
        'name': name,
        'success': success,
        'maze_size': maze_size,
        'steps': step if success else max_steps,
        'total_time': total_time,
        'total_episodes': final_stats['total_episodes'],
        'wall_hits': final_stats['wall_hits'],
        'wall_hit_rate': final_stats['wall_hits'] / max(step, 1),
        'path_length': final_stats['path_length'],
        'distance_to_goal': final_stats['distance_to_goal'],
        'avg_search_time': final_stats['avg_search_time'],
        'depth_usage': final_stats['depth_usage']
    }
    
    # 適応的エージェント特有の統計
    if hasattr(agent, 'stats') and 'adaptive_depth_selections' in agent.stats:
        result['adaptive_depth_selections'] = agent.stats['adaptive_depth_selections']
        result['avg_adaptive_depth'] = final_stats.get('avg_adaptive_depth', 0)
        
        # geDIG改善の分析
        if agent.stats.get('gedig_evaluations'):
            improvements = []
            for eval_history in agent.stats['gedig_evaluations'][:20]:  # 最初の20個
                if len(eval_history) > 1:
                    base = eval_history[0][1]
                    best = min(h[1] for h in eval_history)
                    improvement = (base - best) / (base + 0.001)
                    improvements.append(improvement)
            if improvements:
                result['avg_gedig_improvement'] = np.mean(improvements)
    
    return result


def compare_agents():
    """固定深度と適応的深度を比較"""
    
    print("="*80)
    print("ADAPTIVE geDIG DEPTH SELECTION EXPERIMENT")
    print("Comparing Fixed-depth vs Adaptive-depth Agents")
    print("="*80)
    
    # 実験パラメータ
    maze_sizes = [(7, 7), (11, 11), (15, 15)]
    seeds = [42, 123, 456]
    
    # 結果保存用
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = Path(f"../results/adaptive_comparison_{timestamp}")
    base_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for maze_size in maze_sizes:
        for seed in seeds:
            print(f"\n{'='*80}")
            print(f"Maze: {maze_size[0]}×{maze_size[1]}, Seed: {seed}")
            print(f"{'='*80}")
            
            # 迷路生成
            generator = ProperMazeGenerator()
            maze = generator.generate_dfs_maze(size=maze_size, seed=seed)
            
            # 最大ステップ数
            max_steps = maze_size[0] * maze_size[1] * 10
            
            # 1. 固定深度エージェント（深度3）
            agent_fixed = PureMemoryAgent(
                maze=maze.copy(),
                datastore_path=str(base_path / f"fixed_{maze_size[0]}x{maze_size[1]}_{seed}"),
                config={
                    'max_depth': 3,
                    'search_k': 20
                }
            )
            
            result_fixed = run_single_experiment(
                agent_fixed, 
                maze_size[0], 
                max_steps,
                "Fixed-depth (3-hop)"
            )
            result_fixed['seed'] = seed
            
            # 2. 適応的深度エージェント
            agent_adaptive = PureMemoryAgentAdaptive(
                maze=maze.copy(),
                datastore_path=str(base_path / f"adaptive_{maze_size[0]}x{maze_size[1]}_{seed}"),
                config={
                    'max_depth': 5,
                    'search_k': 20,
                    'gedig_improvement_threshold': 0.05
                }
            )
            
            result_adaptive = run_single_experiment(
                agent_adaptive,
                maze_size[0],
                max_steps,
                "Adaptive-depth (geDIG-based)"
            )
            result_adaptive['seed'] = seed
            
            # 結果を保存
            all_results.append({
                'maze_size': maze_size,
                'seed': seed,
                'fixed': result_fixed,
                'adaptive': result_adaptive
            })
            
            # 比較表示
            print(f"\n{'='*60}")
            print("COMPARISON SUMMARY")
            print(f"{'='*60}")
            
            # 成功率
            print(f"\nSuccess:")
            print(f"  Fixed:    {'✅' if result_fixed['success'] else '❌'}")
            print(f"  Adaptive: {'✅' if result_adaptive['success'] else '❌'}")
            
            # ステップ数
            if result_fixed['success'] or result_adaptive['success']:
                print(f"\nSteps to goal:")
                if result_fixed['success']:
                    print(f"  Fixed:    {result_fixed['steps']}")
                if result_adaptive['success']:
                    print(f"  Adaptive: {result_adaptive['steps']}")
            
            # 壁衝突率
            print(f"\nWall hit rate:")
            print(f"  Fixed:    {result_fixed['wall_hit_rate']:.1%}")
            print(f"  Adaptive: {result_adaptive['wall_hit_rate']:.1%}")
            
            # 深度使用（適応的エージェントのみ）
            if 'avg_adaptive_depth' in result_adaptive:
                print(f"\nAdaptive depth stats:")
                print(f"  Average depth: {result_adaptive['avg_adaptive_depth']:.2f}")
                if 'avg_gedig_improvement' in result_adaptive:
                    print(f"  Avg geDIG improvement: {result_adaptive['avg_gedig_improvement']:.3f}")
    
    # 全体統計
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")
    
    # 成功率集計
    fixed_successes = sum(1 for r in all_results if r['fixed']['success'])
    adaptive_successes = sum(1 for r in all_results if r['adaptive']['success'])
    total_experiments = len(all_results)
    
    print(f"\nSuccess rates:")
    print(f"  Fixed:    {fixed_successes}/{total_experiments} "
          f"({fixed_successes/total_experiments*100:.1f}%)")
    print(f"  Adaptive: {adaptive_successes}/{total_experiments} "
          f"({adaptive_successes/total_experiments*100:.1f}%)")
    
    # ステップ数比較（成功した実験のみ）
    fixed_steps = [r['fixed']['steps'] for r in all_results if r['fixed']['success']]
    adaptive_steps = [r['adaptive']['steps'] for r in all_results if r['adaptive']['success']]
    
    if fixed_steps:
        print(f"\nFixed-depth average steps: {np.mean(fixed_steps):.1f}")
    if adaptive_steps:
        print(f"Adaptive-depth average steps: {np.mean(adaptive_steps):.1f}")
    
    # 壁衝突率比較
    fixed_wall_rates = [r['fixed']['wall_hit_rate'] for r in all_results]
    adaptive_wall_rates = [r['adaptive']['wall_hit_rate'] for r in all_results]
    
    print(f"\nAverage wall hit rates:")
    print(f"  Fixed:    {np.mean(fixed_wall_rates):.1%}")
    print(f"  Adaptive: {np.mean(adaptive_wall_rates):.1%}")
    
    # 適応的深度の分析
    all_adaptive_depths = []
    for r in all_results:
        if 'adaptive_depth_selections' in r['adaptive']:
            all_adaptive_depths.extend(r['adaptive']['adaptive_depth_selections'])
    
    if all_adaptive_depths:
        print(f"\nAdaptive depth distribution:")
        for depth in range(1, 6):
            count = all_adaptive_depths.count(depth)
            if count > 0:
                percentage = count / len(all_adaptive_depths) * 100
                print(f"  {depth}-hop: {count} times ({percentage:.1f}%)")
        print(f"  Average: {np.mean(all_adaptive_depths):.2f}")
    
    # 結果をJSONで保存
    with open(base_path / "comparison_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📁 Results saved to: {base_path}")
    
    return all_results


def main():
    """メイン実行"""
    results = compare_agents()
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    
    # 最終評価
    fixed_successes = sum(1 for r in results if r['fixed']['success'])
    adaptive_successes = sum(1 for r in results if r['adaptive']['success'])
    
    if adaptive_successes > fixed_successes:
        print("✅ Adaptive geDIG-based depth selection shows improvement!")
        print("   The agent successfully adjusts exploration depth based on edge quality.")
    elif adaptive_successes == fixed_successes:
        print("📊 Adaptive and fixed depth perform similarly in success rate.")
        print("   Check other metrics like steps and wall hits for differences.")
    else:
        print("⚠️  Fixed depth performs better in this test.")
        print("   May need to tune geDIG threshold or other parameters.")
    
    # geDIG改善の効果
    gedig_improvements = []
    for r in results:
        if 'avg_gedig_improvement' in r['adaptive']:
            gedig_improvements.append(r['adaptive']['avg_gedig_improvement'])
    
    if gedig_improvements:
        avg_improvement = np.mean(gedig_improvements)
        print(f"\nAverage geDIG improvement through deeper search: {avg_improvement:.3f}")
        if avg_improvement > 0.1:
            print("   ⭐ Significant improvement in edge quality through multi-hop!")
        elif avg_improvement > 0.05:
            print("   📈 Moderate improvement in edge quality")
        else:
            print("   📊 Minimal improvement - may need parameter tuning")


if __name__ == "__main__":
    main()