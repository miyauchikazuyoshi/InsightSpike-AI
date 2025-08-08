#!/usr/bin/env python3
"""
純粋geDIG記憶エージェントの最終テスト
"""

import numpy as np
import time
from datetime import datetime
import sys
import os

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent_final import PureMemoryAgentFinal


def test_pure_final():
    """純粋記憶エージェント最終版のテスト"""
    
    print("="*70)
    print("🎯 純粋geDIG記憶エージェント 最終テスト")
    print("  ・推論結果は破棄")
    print("  ・実際の経験のみ記憶")
    print("  ・純粋な情報理論的評価")
    print("="*70)
    
    # 7×7迷路でテスト
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(7, 7), seed=42)
    
    print("\n迷路 (7×7):")
    for row in maze:
        print(' '.join(['.' if x == 0 else '█' for x in row]))
    
    # エージェント作成
    agent = PureMemoryAgentFinal(
        maze=maze,
        datastore_path="../results/pure_final_test",
        config={
            'max_depth': 3,
            'search_k': 15
        }
    )
    
    print(f"\n📍 スタート: {agent.position}")
    print(f"🎯 ゴール: {agent.goal}")
    
    initial_distance = abs(agent.position[0] - agent.goal[0]) + \
                      abs(agent.position[1] - agent.goal[1])
    print(f"📏 初期距離: {initial_distance}")
    print("-" * 40)
    
    # 実行
    max_steps = 100
    start_time = time.time()
    
    for step in range(max_steps):
        # ゴール確認
        if agent.is_goal_reached():
            elapsed = time.time() - start_time
            stats = agent.get_statistics()
            
            print(f"\n✅ 成功！ {step}ステップでゴール到達")
            print(f"  実行時間: {elapsed:.2f}秒")
            print(f"  壁衝突率: {stats['wall_hit_rate']:.1%}")
            print(f"  総エピソード数: {stats['total_episodes']}")
            
            # メモリ統計
            mem_stats = stats['memory_stats']
            print(f"\n📊 メモリ統計:")
            print(f"  経験数: {mem_stats.get('total_experiences', 0)}")
            print(f"  エッジ数: {mem_stats.get('total_edges', 0)}")
            if 'avg_gedig' in mem_stats:
                print(f"  平均geDIG: {mem_stats['avg_gedig']:.3f}")
            if 'graph_density' in mem_stats:
                print(f"  グラフ密度: {mem_stats['graph_density']:.3f}")
            
            # 深度使用
            print(f"\n深度使用:")
            for depth, count in stats['depth_usage'].items():
                if count > 0:
                    print(f"  {depth}ホップ: {count}回")
            
            return True
        
        # 行動実行
        action = agent.get_action()
        success = agent.execute_action(action)
        
        # 進捗報告
        if step % 20 == 0 and step > 0:
            stats = agent.get_statistics()
            distance = stats['distance_to_goal']
            improvement = (initial_distance - distance) / initial_distance * 100
            
            print(f"\nStep {step}: ")
            print(f"  位置: {stats['position']}")
            print(f"  距離: {distance} ({improvement:+.1f}%改善)")
            print(f"  壁衝突率: {stats['wall_hit_rate']:.1%}")
            print(f"  推論数: {stats['memory_stats'].get('inference_count', 0)}")
    
    # タイムアウト
    elapsed = time.time() - start_time
    final_stats = agent.get_statistics()
    
    print(f"\n⏱️ {max_steps}ステップで未到達")
    print(f"  最終距離: {final_stats['distance_to_goal']}")
    print(f"  壁衝突率: {final_stats['wall_hit_rate']:.1%}")
    print(f"  実行時間: {elapsed:.2f}秒")
    
    return False


def analyze_memory_structure(agent):
    """メモリ構造を分析"""
    print("\n" + "="*70)
    print("🔍 メモリ構造分析")
    print("="*70)
    
    stats = agent.memory.get_statistics()
    
    # グラフ構造
    if 'graph_nodes' in stats:
        print(f"\nグラフ構造:")
        print(f"  ノード数: {stats['graph_nodes']}")
        print(f"  エッジ数: {stats['graph_edges']}")
        print(f"  平均次数: {stats.get('avg_degree', 0):.2f}")
        print(f"  最大次数: {stats.get('max_degree', 0)}")
    
    # geDIG分布
    if 'min_gedig' in stats:
        print(f"\ngeDIG値分布:")
        print(f"  最小: {stats['min_gedig']:.3f}")
        print(f"  平均: {stats['avg_gedig']:.3f}")
        print(f"  中央値: {stats['median_gedig']:.3f}")
        print(f"  最大: {stats['max_gedig']:.3f}")
    
    # 推論統計
    print(f"\n推論統計:")
    print(f"  推論回数: {stats.get('inference_count', 0)}")
    print(f"  （推論結果は全て破棄）")
    
    # メモリ効率
    total_exp = stats.get('total_experiences', 0)
    total_edges = stats.get('total_edges', 0)
    if total_exp > 0:
        edge_ratio = total_edges / total_exp
        print(f"\nメモリ効率:")
        print(f"  エピソード当たりエッジ数: {edge_ratio:.2f}")
        print(f"  メモリ使用量: 実経験のみ（推論結果は保存しない）")


if __name__ == "__main__":
    print("🚀 純粋記憶駆動AI 最終実験")
    print("  報酬なし、強化なし、純粋な情報理論的学習")
    print()
    
    success = test_pure_final()
    
    print("\n" + "="*70)
    if success:
        print("🏆 純粋記憶エージェントが成功！")
        print("   geDIGが評価関数として機能")
        print("   推論と記憶の分離が成功")
    else:
        print("📊 学習継続中")
        print("   より長い学習で改善の可能性")
    print("="*70)