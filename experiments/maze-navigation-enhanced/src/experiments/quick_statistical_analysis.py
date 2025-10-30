#!/usr/bin/env python3
"""
既存データでの統計分析＋迅速な追加実験
"""

import os
import sys
import json
import time
import numpy as np
import random
from datetime import datetime
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from navigation.maze_navigator import MazeNavigator


def run_quick_trials(n_trials=10):
    """迅速な試行（タイムアウト対策）"""
    
    print("迅速な統計実験（N=10）")
    print("=" * 60)
    
    results = {'simple': [], 'gedig': []}
    
    for trial in range(n_trials):
        seed = 2000 + trial
        random.seed(seed)
        np.random.seed(seed)
        
        # シンプルな迷路（15x15）
        maze = np.ones((15, 15), dtype=int)
        
        # 基本的な通路を作成
        maze[1, 1:14] = 0  # 上の通路
        maze[1:14, 13] = 0  # 右の通路
        maze[13, 1:14] = 0  # 下の通路
        maze[1:14, 1] = 0  # 左の通路
        
        # 中央に十字路
        maze[7, 3:12] = 0
        maze[3:12, 7] = 0
        
        # いくつかランダムな通路
        for _ in range(10):
            x = random.randint(2, 12)
            y = random.randint(2, 12)
            if random.random() > 0.5:
                maze[y, max(1, x-1):min(14, x+2)] = 0
            else:
                maze[max(1, y-1):min(14, y+2), x] = 0
        
        print(f"試行 {trial+1}/{n_trials}: ", end='', flush=True)
        
        for strategy in ['simple', 'gedig']:
            start = (1, 1)
            goal = (13, 13)
            
            nav = MazeNavigator(
                maze=maze,
                start_pos=start,
                goal_pos=goal,
                wiring_strategy=strategy,
                gedig_threshold=-0.08,
                backtrack_threshold=-0.2,
                simple_mode=True
            )
            
            start_time = time.perf_counter()
            path = []
            
            for step in range(500):  # 短いタイムアウト
                action = nav.step()
                path.append(nav.current_pos)
                
                if nav.current_pos == goal:
                    break
            
            elapsed = time.perf_counter() - start_time
            graph_stats = nav.graph_manager.get_graph_statistics()
            
            result = {
                'success': nav.current_pos == goal,
                'steps': len(path),
                'edges': graph_stats['num_edges'],
                'time': elapsed
            }
            
            results[strategy].append(result)
            
            status = '✓' if result['success'] else '✗'
            print(f"{strategy}={status}({result['steps']})", end=' ')
        
        print()
    
    return results


def analyze_results(results):
    """結果の統計分析"""
    
    print("\n" + "=" * 60)
    print("統計分析結果")
    print("=" * 60)
    
    # ステップ数の分析
    simple_steps = [r['steps'] for r in results['simple'] if r['success']]
    gedig_steps = [r['steps'] for r in results['gedig'] if r['success']]
    
    if len(simple_steps) > 0 and len(gedig_steps) > 0:
        print("\n【ステップ数】")
        print(f"  Simple: {np.mean(simple_steps):.1f} ± {np.std(simple_steps):.1f}")
        print(f"  geDIG:  {np.mean(gedig_steps):.1f} ± {np.std(gedig_steps):.1f}")
        
        # t検定
        t_stat, p_value = stats.ttest_ind(simple_steps, gedig_steps)
        print(f"  t = {t_stat:.3f}, p = {p_value:.4f}")
        
        if p_value < 0.05:
            print("  → 統計的に有意な差あり！")
        
        # 改善率
        improvement = (np.mean(simple_steps) - np.mean(gedig_steps)) / np.mean(simple_steps) * 100
        print(f"  改善率: {improvement:+.1f}%")
    
    # エッジ数の分析
    simple_edges = [r['edges'] for r in results['simple']]
    gedig_edges = [r['edges'] for r in results['gedig']]
    
    print("\n【エッジ数】")
    print(f"  Simple: {np.mean(simple_edges):.1f} ± {np.std(simple_edges):.1f}")
    print(f"  geDIG:  {np.mean(gedig_edges):.1f} ± {np.std(gedig_edges):.1f}")
    
    edge_reduction = (np.mean(simple_edges) - np.mean(gedig_edges)) / np.mean(simple_edges) * 100
    print(f"  削減率: {edge_reduction:.1f}%")
    
    # 成功率
    simple_success = sum(1 for r in results['simple'] if r['success']) / len(results['simple'])
    gedig_success = sum(1 for r in results['gedig'] if r['success']) / len(results['gedig'])
    
    print("\n【成功率】")
    print(f"  Simple: {simple_success*100:.1f}%")
    print(f"  geDIG:  {gedig_success*100:.1f}%")


def generate_paper_table(results):
    """論文用の表を生成"""
    
    print("\n" + "=" * 60)
    print("論文用表（LaTeX形式）")
    print("=" * 60)
    
    simple_steps = [r['steps'] for r in results['simple'] if r['success']]
    gedig_steps = [r['steps'] for r in results['gedig'] if r['success']]
    simple_edges = [r['edges'] for r in results['simple'] if r['success']]
    gedig_edges = [r['edges'] for r in results['gedig'] if r['success']]
    
    print("""
\\begin{table}[h]
\\centering
\\caption{Maze Navigation Performance Comparison (N=%d)}
\\begin{tabular}{lcccc}
\\hline
Strategy & Steps & Edges & Success Rate & p-value \\\\
\\hline
Simple & %.1f ± %.1f & %.1f ± %.1f & %.1f\\%% & - \\\\
geDIG (NoCopy) & %.1f ± %.1f & %.1f ± %.1f & %.1f\\%% & %.4f \\\\
\\hline
Improvement & %.1f\\%% & %.1f\\%% & - & - \\\\
\\hline
\\end{tabular}
\\end{table}
""" % (
        len(results['simple']),
        np.mean(simple_steps), np.std(simple_steps),
        np.mean(simple_edges), np.std(simple_edges),
        sum(1 for r in results['simple'] if r['success']) / len(results['simple']) * 100,
        np.mean(gedig_steps), np.std(gedig_steps),
        np.mean(gedig_edges), np.std(gedig_edges),
        sum(1 for r in results['gedig'] if r['success']) / len(results['gedig']) * 100,
        stats.ttest_ind(simple_steps, gedig_steps)[1],
        (np.mean(simple_steps) - np.mean(gedig_steps)) / np.mean(simple_steps) * 100,
        (np.mean(simple_edges) - np.mean(gedig_edges)) / np.mean(simple_edges) * 100
    ))


if __name__ == '__main__':
    # 迅速な実験を実行
    results = run_quick_trials(n_trials=10)
    
    # 分析
    analyze_results(results)
    
    # 論文用の表を生成
    generate_paper_table(results)
    
    # 結果を保存
    os.makedirs('results/statistical_quick', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'results/statistical_quick/results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n結果を保存: results/statistical_quick/results_{timestamp}.json")