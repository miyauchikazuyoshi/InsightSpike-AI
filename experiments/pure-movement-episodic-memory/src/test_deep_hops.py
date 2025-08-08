#!/usr/bin/env python3
"""
深いホップ数での実験
エッジ数7でも深い推論で補えるか検証
"""

import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from pure_memory_agent_optimized import PureMemoryAgentOptimized
from test_true_perfect_maze import generate_perfect_maze_dfs


def test_with_depth_and_edges(max_depth, edge_count, maze, trial_seed=None):
    """指定深度とエッジ数でテスト"""
    
    # エージェント作成
    agent = PureMemoryAgentOptimized(
        maze=maze,
        datastore_path=f"../results/depth_{max_depth}_edge_{edge_count}",
        config={
            'max_depth': max_depth,  # ホップ数
            'search_k': 30,
            'gedig_threshold': 0.5,
            'max_edges_per_node': edge_count  # エッジ数
        }
    )
    
    # 実行
    path = [agent.position]
    depth_usage = {i: 0 for i in range(1, max_depth+1)}
    start_time = time.time()
    
    for step in range(300):  # より長い時間許可
        if agent.is_goal_reached():
            elapsed = time.time() - start_time
            
            # 深度使用統計を取得
            for d, count in agent.stats['depth_usage'].items():
                if d in depth_usage:
                    depth_usage[d] = count
            
            return {
                'success': True,
                'steps': step,
                'time': elapsed,
                'path_length': len(path),
                'unique_visits': len(set(path)),
                'wall_hits': agent.stats['wall_hits'],
                'graph_edges': agent.experience_graph.number_of_edges(),
                'depth_usage': depth_usage,
                'avg_search_time': np.mean(agent.stats['search_times']) if agent.stats['search_times'] else 0
            }
        
        action = agent.get_action()
        agent.execute_action(action)
        path.append(agent.position)
    
    # 深度使用統計
    for d, count in agent.stats['depth_usage'].items():
        if d in depth_usage:
            depth_usage[d] = count
    
    return {
        'success': False,
        'steps': 300,
        'time': time.time() - start_time,
        'path_length': len(path),
        'unique_visits': len(set(path)),
        'wall_hits': agent.stats['wall_hits'],
        'graph_edges': agent.experience_graph.number_of_edges(),
        'depth_usage': depth_usage,
        'avg_search_time': np.mean(agent.stats['search_times']) if agent.stats['search_times'] else 0
    }


def run_deep_hop_experiment():
    """深いホップ数の実験"""
    
    print("="*70)
    print("🧠 深いホップ数実験: エッジ数7での深度補償")
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
    
    # 実験1: エッジ数固定（7）、深度を変える
    print("\n" + "="*70)
    print("📊 実験1: エッジ数7固定、深度を変化")
    print("="*70)
    
    depths = [3, 5, 7, 10, 15, 20]
    results_depth = {}
    
    for depth in depths:
        print(f"\n【深度: {depth}ホップ】")
        
        # 3回試行
        trials = []
        for trial in range(3):
            result = test_with_depth_and_edges(depth, 7, maze, trial)
            trials.append(result)
            
            if result['success']:
                print(f"  試行{trial+1}: ✅ {result['steps']}ステップ ({result['time']:.1f}秒)")
            else:
                print(f"  試行{trial+1}: ❌ 失敗 ({result['time']:.1f}秒)")
        
        # 統計
        success_rate = sum(1 for r in trials if r['success']) / len(trials)
        avg_steps = np.mean([r['steps'] for r in trials if r['success']]) if success_rate > 0 else 0
        avg_time = np.mean([r['time'] for r in trials])
        avg_search = np.mean([r['avg_search_time'] for r in trials])
        
        results_depth[depth] = {
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'avg_time': avg_time,
            'avg_search_ms': avg_search
        }
        
        print(f"  成功率: {success_rate*100:.0f}%")
        print(f"  平均検索時間: {avg_search:.2f}ms/クエリ")
    
    # 実験2: 深度とエッジ数の組み合わせ
    print("\n" + "="*70)
    print("📊 実験2: 深度×エッジ数の組み合わせ比較")
    print("="*70)
    
    combinations = [
        (5, 7, "浅い推論×少エッジ"),
        (10, 7, "深い推論×少エッジ"),
        (5, 15, "浅い推論×多エッジ"),
        (10, 15, "深い推論×多エッジ"),
    ]
    
    results_combo = {}
    
    for depth, edges, desc in combinations:
        print(f"\n【{desc}】深度{depth}、エッジ{edges}")
        
        result = test_with_depth_and_edges(depth, edges, maze, 42)
        results_combo[(depth, edges)] = result
        
        if result['success']:
            print(f"  結果: ✅ {result['steps']}ステップ")
        else:
            print(f"  結果: ❌ 失敗")
        
        # 深度使用分布
        if result['depth_usage']:
            total_usage = sum(result['depth_usage'].values())
            if total_usage > 0:
                print(f"  深度使用分布:")
                for d in range(1, min(6, depth+1)):
                    usage = result['depth_usage'].get(d, 0)
                    if usage > 0:
                        bar = '█' * int(usage/total_usage * 20)
                        print(f"    {d}ホップ: {bar} {usage/total_usage*100:.1f}%")
    
    # 結果分析
    print("\n" + "="*70)
    print("📈 結果分析")
    print("="*70)
    
    print("\n【エッジ数7での深度効果】")
    print("深度 | 成功率 | 平均ステップ | 平均時間 | 検索速度")
    print("-" * 60)
    
    for depth in depths:
        r = results_depth[depth]
        print(f" {depth:2d}  | {r['success_rate']*100:5.0f}% | "
              f"{r['avg_steps']:7.0f}    | {r['avg_time']:6.2f}秒 | "
              f"{r['avg_search_ms']:5.2f}ms")
    
    print("\n【組み合わせ比較】")
    print("深度×エッジ | 成功 | ステップ数 | 実行時間")
    print("-" * 50)
    
    for (depth, edges), result in results_combo.items():
        success = "✅" if result['success'] else "❌"
        print(f" {depth:2d} × {edges:2d}   |  {success}  | "
              f"{result['steps']:6d}   | {result['time']:6.2f}秒")
    
    # 洞察
    print("\n" + "="*70)
    print("💡 洞察")
    print("="*70)
    
    # 深度10以上でエッジ7が成功するか確認
    deep_success = any(results_depth[d]['success_rate'] > 0 for d in depths if d >= 10)
    
    if deep_success:
        print("\n✨ 深い推論がエッジ数の少なさを補償！")
        print("  - マジカルナンバー7でも深い推論で解決可能")
        print("  - グラフは疎でも、多段階推論で経路発見")
    else:
        print("\n⚠️ 深い推論だけでは不十分")
        print("  - 基本的な連結性（エッジ数）が重要")
        print("  - 深度を増やしても疎グラフでは限界")
    
    print("\n🧠 認知科学的解釈:")
    print("  - 人間も「深く考える」ことで記憶の制約を補う")
    print("  - 7±2の制約は「同時」処理の限界")
    print("  - 逐次的な深い推論で複雑な問題を解決")
    
    # 最適な組み合わせを提案
    print("\n📝 推奨設定:")
    if deep_success:
        optimal_depth = min(d for d in depths if d >= 10 and results_depth[d]['success_rate'] > 0)
        print(f"  - エッジ数: 7（マジカルナンバー）")
        print(f"  - 推論深度: {optimal_depth}ホップ")
        print(f"  - 根拠: 認知的妥当性を保ちつつ性能確保")
    else:
        print(f"  - エッジ数: 10-15（実用的な最小値）")
        print(f"  - 推論深度: 5-7ホップ")
        print(f"  - 根拠: 安定した探索性能を優先")


if __name__ == "__main__":
    # 結果ディレクトリ作成
    os.makedirs('../results', exist_ok=True)
    
    # 実験実行
    run_deep_hop_experiment()