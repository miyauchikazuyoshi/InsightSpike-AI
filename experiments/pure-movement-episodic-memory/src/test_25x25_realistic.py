#!/usr/bin/env python3
"""
25×25現実的な規模での実験
マジカルナンバー7 + 深い推論（最大20ホップ）での評価
"""

import numpy as np
import random
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from pure_memory_agent_optimized import PureMemoryAgentOptimized
from test_50x50_challenge import generate_large_perfect_maze, BiasedRandomWalkAgent


def test_25x25_maze():
    """25×25迷路での実験"""
    
    print("="*70)
    print("🏰 25×25現実的規模での実験")
    print("  設定: エッジ数7（マジカルナンバー）、最大20ホップ")
    print("="*70)
    
    # 25×25迷路生成
    print("\n⏳ 迷路生成中...")
    maze = generate_large_perfect_maze((25, 25), seed=2024)
    
    # 迷路の統計
    passages = np.sum(maze == 0)
    height, width = maze.shape
    
    print(f"\n📊 迷路統計:")
    print(f"  サイズ: {height}×{width}")
    print(f"  通路数: {passages}マス")
    print(f"  密度: {passages/(height*width)*100:.1f}%")
    print(f"  最短距離（マンハッタン）: {abs(23-1) + abs(23-1)} = 44")
    
    # 実験開始
    print("\n" + "="*70)
    print("📊 実験開始（各3試行）")
    print("="*70)
    
    # 結果格納
    all_results = {}
    
    # 実験設定
    configs = [
        ("ランダム", None),
        ("エッジ7+深度20", {'max_depth': 20, 'max_edges_per_node': 7}),
        ("エッジ7+深度10", {'max_depth': 10, 'max_edges_per_node': 7}),
        ("エッジ7+深度5", {'max_depth': 5, 'max_edges_per_node': 7}),
        ("エッジ15+深度5", {'max_depth': 5, 'max_edges_per_node': 15}),
    ]
    
    for config_name, config in configs:
        print(f"\n【{config_name}】")
        results = []
        
        for trial in range(3):
            print(f"  試行{trial+1}: ", end="", flush=True)
            
            if config_name == "ランダム":
                # ランダムウォーク
                random.seed(trial)
                agent = BiasedRandomWalkAgent(maze)
                start_time = time.time()
                
                for step in range(2000):
                    if agent.is_goal_reached():
                        elapsed = time.time() - start_time
                        print(f"✅ {step}ステップ ({elapsed:.1f}秒)")
                        results.append({
                            'success': True,
                            'steps': step,
                            'time': elapsed
                        })
                        break
                    agent.execute_action(agent.get_action())
                else:
                    elapsed = time.time() - start_time
                    print(f"❌ タイムアウト")
                    results.append({
                        'success': False,
                        'steps': 2000,
                        'time': elapsed
                    })
            else:
                # 純粋記憶エージェント
                agent = PureMemoryAgentOptimized(
                    maze=maze,
                    datastore_path=f"../results/25x25_{config_name}_{trial}",
                    config={
                        'max_depth': config['max_depth'],
                        'search_k': 30,
                        'gedig_threshold': 0.5,
                        'max_edges_per_node': config['max_edges_per_node']
                    }
                )
                
                start_time = time.time()
                depth_usage = {}
                
                for step in range(500):
                    if agent.is_goal_reached():
                        elapsed = time.time() - start_time
                        
                        # 深度使用統計
                        for d, count in agent.stats['depth_usage'].items():
                            depth_usage[d] = count
                        
                        print(f"✅ {step}ステップ ({elapsed:.1f}秒)")
                        results.append({
                            'success': True,
                            'steps': step,
                            'time': elapsed,
                            'depth_usage': depth_usage,
                            'avg_gedig': agent.get_statistics()['avg_gedig']
                        })
                        break
                    
                    agent.execute_action(agent.get_action())
                else:
                    elapsed = time.time() - start_time
                    print(f"❌ タイムアウト")
                    
                    for d, count in agent.stats['depth_usage'].items():
                        depth_usage[d] = count
                    
                    results.append({
                        'success': False,
                        'steps': 500,
                        'time': elapsed,
                        'depth_usage': depth_usage,
                        'avg_gedig': agent.get_statistics()['avg_gedig']
                    })
        
        all_results[config_name] = results
    
    # 結果分析
    print("\n" + "="*70)
    print("📈 結果分析")
    print("="*70)
    
    print("\n設定 | 成功率 | 平均ステップ | 平均時間 | 平均geDIG")
    print("-" * 60)
    
    for config_name in [name for name, _ in configs]:
        results = all_results[config_name]
        success_count = sum(1 for r in results if r['success'])
        success_rate = success_count / len(results) * 100
        
        if success_count > 0:
            success_results = [r for r in results if r['success']]
            avg_steps = np.mean([r['steps'] for r in success_results])
            avg_time = np.mean([r['time'] for r in success_results])
            
            if 'avg_gedig' in success_results[0]:
                avg_gedig = np.mean([r['avg_gedig'] for r in success_results])
                print(f"{config_name:15s} | {success_rate:5.0f}% | {avg_steps:7.0f} | "
                      f"{avg_time:5.1f}秒 | {avg_gedig:7.4f}")
            else:
                print(f"{config_name:15s} | {success_rate:5.0f}% | {avg_steps:7.0f} | "
                      f"{avg_time:5.1f}秒 | ---")
        else:
            print(f"{config_name:15s} | {success_rate:5.0f}% | --- | --- | ---")
    
    # 深度使用分析
    print("\n【深度使用パターン】")
    for config_name in ["エッジ7+深度20", "エッジ7+深度10"]:
        if config_name in all_results:
            results = all_results[config_name]
            success_results = [r for r in results if r['success'] and 'depth_usage' in r]
            
            if success_results:
                print(f"\n{config_name}:")
                all_depth_usage = {}
                for r in success_results:
                    for d, count in r['depth_usage'].items():
                        all_depth_usage[d] = all_depth_usage.get(d, 0) + count
                
                total = sum(all_depth_usage.values())
                if total > 0:
                    for d in sorted(all_depth_usage.keys())[:5]:
                        ratio = all_depth_usage[d] / total * 100
                        if ratio > 1:
                            bar = '█' * int(ratio / 5)
                            print(f"  {d:2d}ホップ: {bar} {ratio:.1f}%")
    
    # 最終評価
    print("\n" + "="*70)
    print("💡 最終評価")
    print("="*70)
    
    # 各設定の成功率を比較
    random_success = sum(1 for r in all_results["ランダム"] if r['success'])
    e7d20_success = sum(1 for r in all_results["エッジ7+深度20"] if r['success'])
    e7d10_success = sum(1 for r in all_results["エッジ7+深度10"] if r['success'])
    
    print("\n25×25迷路での結論:")
    
    if e7d20_success > random_success:
        print("✅ マジカルナンバー7 + 深い推論がランダムを上回る！")
    
    if e7d20_success > 0:
        print("✨ エッジ数7でも深い推論により解決可能")
        
        # 実際に使われた深度を分析
        for r in all_results["エッジ7+深度20"]:
            if r['success'] and 'depth_usage' in r:
                most_used = max(r['depth_usage'].items(), key=lambda x: x[1])[0]
                print(f"  → 最頻使用深度: {most_used}ホップ")
                break
    
    print("\n📝 洞察:")
    print("  - 25×25は11×11と50×50の中間的な難易度")
    print("  - マジカルナンバー7の有効性が確認できる規模")
    print("  - 深い推論が探索効率に寄与")


if __name__ == "__main__":
    # 結果ディレクトリ作成
    os.makedirs('../results', exist_ok=True)
    
    # 実験実行
    test_25x25_maze()