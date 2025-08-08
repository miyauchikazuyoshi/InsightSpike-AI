#!/usr/bin/env python3
"""
マジカルナンバー7でのエッジ数実験
人間の認知限界に合わせた設定での性能評価
"""

import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from pure_memory_agent_optimized import PureMemoryAgentOptimized
from test_true_perfect_maze import generate_perfect_maze_dfs


def test_with_edge_count(edge_count, maze, seed=None):
    """指定エッジ数でテスト"""
    
    # エージェント作成
    agent = PureMemoryAgentOptimized(
        maze=maze,
        datastore_path=f"../results/edge_{edge_count}",
        config={
            'max_depth': 5,
            'search_k': 30,
            'gedig_threshold': 0.5,
            'max_edges_per_node': edge_count  # エッジ数を指定
        }
    )
    
    # 実行
    path = [agent.position]
    start_time = time.time()
    
    for step in range(200):
        if agent.is_goal_reached():
            elapsed = time.time() - start_time
            return {
                'success': True,
                'steps': step,
                'time': elapsed,
                'path_length': len(path),
                'unique_visits': len(set(path)),
                'wall_hits': agent.stats['wall_hits'],
                'graph_edges': agent.experience_graph.number_of_edges()
            }
        
        action = agent.get_action()
        agent.execute_action(action)
        path.append(agent.position)
    
    return {
        'success': False,
        'steps': 200,
        'time': time.time() - start_time,
        'path_length': len(path),
        'unique_visits': len(set(path)),
        'wall_hits': agent.stats['wall_hits'],
        'graph_edges': agent.experience_graph.number_of_edges()
    }


def run_comparison():
    """エッジ数による性能比較"""
    
    print("="*70)
    print("🧠 マジカルナンバー実験: エッジ数による性能変化")
    print("="*70)
    
    # 11×11の完全迷路を生成
    maze = generate_perfect_maze_dfs((11, 11), seed=42)
    
    print("\n実験迷路（11×11完全迷路）:")
    for i, row in enumerate(maze):
        row_str = ""
        for j, cell in enumerate(row):
            if i == 1 and j == 1:
                row_str += "S"
            elif i == 9 and j == 9:
                row_str += "G"
            elif cell == 1:
                row_str += "█"
            else:
                row_str += " "
        print(row_str)
    
    # 異なるエッジ数でテスト
    edge_counts = [3, 5, 7, 10, 15, 20]
    results = {}
    
    print("\n" + "="*70)
    print("📊 各エッジ数での性能測定（3回平均）")
    print("="*70)
    
    for edge_count in edge_counts:
        print(f"\n【エッジ数: {edge_count}】")
        
        # 3回試行の平均
        trials = []
        for trial in range(3):
            result = test_with_edge_count(edge_count, maze, seed=trial)
            trials.append(result)
            
            if result['success']:
                print(f"  試行{trial+1}: ✅ {result['steps']}ステップ")
            else:
                print(f"  試行{trial+1}: ❌ 失敗")
        
        # 統計計算
        success_rate = sum(1 for r in trials if r['success']) / len(trials)
        avg_steps = np.mean([r['steps'] for r in trials if r['success']]) if success_rate > 0 else 0
        avg_time = np.mean([r['time'] for r in trials])
        avg_edges = np.mean([r['graph_edges'] for r in trials])
        
        results[edge_count] = {
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'avg_time': avg_time,
            'avg_edges': avg_edges
        }
        
        print(f"  成功率: {success_rate*100:.0f}%")
        if avg_steps > 0:
            print(f"  平均ステップ: {avg_steps:.0f}")
        print(f"  平均時間: {avg_time:.2f}秒")
        print(f"  平均エッジ数: {avg_edges:.0f}")
    
    # 結果分析
    print("\n" + "="*70)
    print("📈 結果分析")
    print("="*70)
    
    print("\n比較表:")
    print("エッジ数 | 成功率 | 平均ステップ | 平均時間 | グラフサイズ")
    print("-" * 60)
    
    for edge_count in edge_counts:
        r = results[edge_count]
        print(f"  {edge_count:2d}    | {r['success_rate']*100:5.0f}% | "
              f"{r['avg_steps']:7.0f}    | {r['avg_time']:6.2f}秒 | "
              f"{r['avg_edges']:6.0f}")
    
    # マジカルナンバー7の評価
    print("\n" + "="*70)
    print("💡 マジカルナンバー7の評価")
    print("="*70)
    
    r7 = results[7]
    r10 = results[10]
    
    print(f"\nエッジ数7 vs 10（現在のデフォルト）:")
    print(f"  成功率: {r7['success_rate']*100:.0f}% vs {r10['success_rate']*100:.0f}%")
    
    if r7['avg_steps'] > 0 and r10['avg_steps'] > 0:
        step_diff = (r7['avg_steps'] - r10['avg_steps']) / r10['avg_steps'] * 100
        print(f"  ステップ数: {'+' if step_diff > 0 else ''}{step_diff:.1f}%")
    
    time_diff = (r7['avg_time'] - r10['avg_time']) / r10['avg_time'] * 100
    print(f"  実行時間: {'+' if time_diff > 0 else ''}{time_diff:.1f}%")
    
    edge_diff = (r7['avg_edges'] - r10['avg_edges']) / r10['avg_edges'] * 100
    print(f"  グラフサイズ: {edge_diff:.1f}%削減")
    
    print("\n🧠 認知科学的解釈:")
    print("  - 7エッジは人間の作業記憶容量に適合")
    print("  - 関連エピソードの同時考慮が可能な範囲")
    print("  - メモリ効率と探索性能のバランス点")
    
    # 最適エッジ数の推奨
    best_edge = max(results.keys(), 
                    key=lambda k: results[k]['success_rate'] * 100 - 
                                 results[k]['avg_time'] * 10)
    
    print(f"\n推奨エッジ数: {best_edge}")
    if best_edge == 7:
        print("  → マジカルナンバー7が最適！ ✨")
    else:
        print(f"  → 性能面では{best_edge}が最適だが、")
        print("    認知的妥当性を考慮すると7も有力な選択肢")


if __name__ == "__main__":
    # 結果ディレクトリ作成
    os.makedirs('../results', exist_ok=True)
    
    # 実験実行
    run_comparison()