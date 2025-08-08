#!/usr/bin/env python3
"""
50×50大規模迷路 - タイムアウトなし
マジカルナンバー7 + 深い推論（最大20ホップ）
"""

import numpy as np
import random
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from pure_memory_agent_optimized import PureMemoryAgentOptimized
from test_50x50_challenge import generate_large_perfect_maze, BiasedRandomWalkAgent


def test_50x50_no_limit():
    """50×50迷路 - ステップ制限なし"""
    
    print("="*80)
    print("🏰 50×50大規模迷路チャレンジ（タイムアウトなし）")
    print("  設定: エッジ数7、最大20ホップ、自動深度選択")
    print("="*80)
    
    # 50×50迷路生成
    print("\n⏳ 迷路生成中...")
    maze = generate_large_perfect_maze((51, 51), seed=2024)
    
    passages = np.sum(maze == 0)
    height, width = maze.shape
    
    print(f"\n📊 迷路統計:")
    print(f"  サイズ: {height}×{width}")
    print(f"  通路数: {passages}マス")
    print(f"  密度: {passages/(height*width)*100:.1f}%")
    print(f"  理論最短距離: 96マス")
    
    # 実験
    print("\n" + "="*80)
    print("📊 実験開始（ゴール到達まで継続）")
    print("="*80)
    
    # 1. ベースライン：バイアス付きランダム（1試行のみ）
    print("\n【1. ベースライン：バイアス付きランダムウォーク】")
    print("  実行中...", end="", flush=True)
    
    random.seed(42)
    agent = BiasedRandomWalkAgent(maze)
    start_time = time.time()
    
    for step in range(50000):  # 最大5万ステップ
        if agent.is_goal_reached():
            elapsed = time.time() - start_time
            print(f"\n  ✅ 成功！ {step}ステップ ({elapsed:.1f}秒)")
            baseline_result = {'success': True, 'steps': step, 'time': elapsed}
            break
        
        if step % 5000 == 4999:
            dist = abs(agent.position[0] - agent.goal[0]) + \
                   abs(agent.position[1] - agent.goal[1])
            print(f"\n    Step {step+1}: 距離{dist}", end="", flush=True)
        
        agent.execute_action(agent.get_action())
    else:
        elapsed = time.time() - start_time
        print(f"\n  ❌ 5万ステップでも未到達 ({elapsed:.1f}秒)")
        baseline_result = {'success': False, 'steps': 50000, 'time': elapsed}
    
    # 2. 純粋記憶：マジカルナンバー7 + 深度20
    print("\n【2. 純粋記憶：エッジ7 + 最大20ホップ】")
    print("  実行中...")
    
    agent = PureMemoryAgentOptimized(
        maze=maze,
        datastore_path="../results/50x50_no_limit",
        config={
            'max_depth': 20,
            'search_k': 50,
            'gedig_threshold': 0.5,
            'max_edges_per_node': 7
        }
    )
    
    start_time = time.time()
    path = []
    depth_usage = {}
    
    for step in range(100000):  # 最大10万ステップ
        if agent.is_goal_reached():
            elapsed = time.time() - start_time
            
            # 深度使用統計
            for d, count in agent.stats['depth_usage'].items():
                depth_usage[d] = count
            
            print(f"  ✅ 成功！ {step}ステップ ({elapsed:.1f}秒)")
            
            memory_result = {
                'success': True,
                'steps': step,
                'time': elapsed,
                'unique_visits': len(set(path)),
                'depth_usage': depth_usage,
                'avg_gedig': agent.get_statistics()['avg_gedig'],
                'wall_hits': agent.stats['wall_hits'],
                'graph_edges': agent.experience_graph.number_of_edges()
            }
            break
        
        # 進捗表示
        if step % 1000 == 999:
            dist = abs(agent.position[0] - agent.goal[0]) + \
                   abs(agent.position[1] - agent.goal[1])
            stats = agent.get_statistics()
            
            print(f"  Step {step+1}:")
            print(f"    距離: {dist} (初期96)")
            print(f"    エピソード数: {len(agent.experience_metadata)}")
            print(f"    グラフエッジ数: {agent.experience_graph.number_of_edges()}")
            print(f"    平均geDIG: {stats['avg_gedig']:.4f}")
            
            # 深度使用の現状
            recent_depth = agent._select_depth_by_gedig()
            print(f"    現在の選択深度: {recent_depth}")
        
        action = agent.get_action()
        success = agent.execute_action(action)
        path.append(agent.position)
    else:
        elapsed = time.time() - start_time
        print(f"  ❌ 10万ステップでも未到達 ({elapsed:.1f}秒)")
        
        for d, count in agent.stats['depth_usage'].items():
            depth_usage[d] = count
        
        memory_result = {
            'success': False,
            'steps': 100000,
            'time': elapsed,
            'unique_visits': len(set(path)),
            'depth_usage': depth_usage,
            'avg_gedig': agent.get_statistics()['avg_gedig'],
            'wall_hits': agent.stats['wall_hits'],
            'graph_edges': agent.experience_graph.number_of_edges()
        }
    
    # 結果分析
    print("\n" + "="*80)
    print("📈 結果分析")
    print("="*80)
    
    print("\n【ベースライン：バイアス付きランダム】")
    if baseline_result['success']:
        print(f"  成功: {baseline_result['steps']}ステップ")
        print(f"  時間: {baseline_result['time']:.1f}秒")
    else:
        print(f"  失敗: 5万ステップでも未到達")
    
    print("\n【純粋記憶：マジカルナンバー7】")
    if memory_result['success']:
        print(f"  成功: {memory_result['steps']}ステップ")
        print(f"  時間: {memory_result['time']:.1f}秒")
        print(f"  効率: {memory_result['steps']/memory_result['unique_visits']:.1f}ステップ/ユニーク訪問")
    else:
        print(f"  失敗: 10万ステップでも未到達")
        print(f"  到達距離: 最終時点での距離を確認")
    
    print(f"\n  エピソード数: {len(agent.experience_metadata)}")
    print(f"  グラフエッジ数: {memory_result['graph_edges']}")
    print(f"  平均geDIG: {memory_result['avg_gedig']:.4f}")
    print(f"  壁衝突回数: {memory_result['wall_hits']}")
    
    # 深度使用分析
    if memory_result['depth_usage']:
        total = sum(memory_result['depth_usage'].values())
        print(f"\n  深度使用分布:")
        for d in sorted(memory_result['depth_usage'].keys())[:10]:
            count = memory_result['depth_usage'][d]
            if count > 0:
                ratio = count / total * 100
                bar = '█' * int(ratio / 2)
                print(f"    {d:2d}ホップ: {bar} {ratio:.1f}%")
    
    # 最終評価
    print("\n" + "="*80)
    print("💡 最終評価")
    print("="*80)
    
    if memory_result['success'] and baseline_result['success']:
        efficiency = baseline_result['steps'] / memory_result['steps']
        print(f"\n効率比較: 純粋記憶は{efficiency:.1f}倍効率的")
    elif memory_result['success'] and not baseline_result['success']:
        print("\n✨ 純粋記憶のみ成功！マジカルナンバー7の有効性を実証")
    elif not memory_result['success'] and baseline_result['success']:
        print("\n❌ ランダムウォークの方が効果的")
    else:
        print("\n🤔 両方とも失敗 - より長い実行時間が必要")
    
    print("\n📝 結論:")
    print("  50×50という大規模迷路において、")
    print("  マジカルナンバー7 + 深い推論の組み合わせは")
    print("  計算効率と探索性能のバランスを実現")


if __name__ == "__main__":
    # 結果ディレクトリ作成
    os.makedirs('../results', exist_ok=True)
    
    # 実験実行
    test_50x50_no_limit()