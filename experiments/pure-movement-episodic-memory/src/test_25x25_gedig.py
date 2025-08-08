#!/usr/bin/env python3
"""
25×25迷路でのgeDIGテスト（段階的テスト）
"""

import numpy as np
import time
from typing import Dict

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from pure_gedig_no_cheat import PureGedigNoCheat
from test_true_perfect_maze import generate_perfect_maze_dfs


def test_25x25_maze():
    """25×25迷路でのテスト"""
    print("="*70)
    print("🌟 25×25迷路 geDIG実装テスト")
    print("="*70)
    
    # 25×25迷路生成
    print("\n🔨 25×25迷路を生成中...")
    maze = generate_perfect_maze_dfs((25, 25), seed=42)
    
    print(f"  迷路サイズ: {maze.shape}")
    print(f"  スタート: (1, 1)")
    print(f"  ゴール: (23, 23)")
    print(f"  最短距離（マンハッタン）: {22 + 22} = 44")
    
    # エージェント作成
    print("\n🤖 エージェント初期化...")
    agent = PureGedigNoCheat(
        maze=maze,
        datastore_path="data/maze_25x25_gedig",
        config={
            'max_edges_per_node': 7,   # マジカルナンバー7
            'gedig_threshold': 0.5,
            'max_depth': 20,           # 最大20ホップ
            'search_k': 50
        }
    )
    
    print("\n🏃 実行開始...")
    print("-" * 70)
    
    start_time = time.time()
    max_steps = 5000
    
    for step in range(max_steps):
        if agent.is_goal_reached():
            elapsed = time.time() - start_time
            print(f"\n🎉 成功！")
            print(f"  ステップ数: {step}")
            print(f"  実行時間: {elapsed:.1f}秒")
            break
        
        # 行動決定と実行
        action = agent.get_action()
        success = agent.execute_action(action)
        
        # 定期進捗
        if step > 0 and step % 100 == 0:
            stats = agent.get_statistics()
            print(f"  Step {step}: 位置{agent.position}, 距離{stats['distance_to_goal']}, "
                  f"エピソード{stats['episodes']}, エッジ{stats['edges']}, "
                  f"平均geDIG{stats['avg_gedig']:.3f}")
    
    else:
        print(f"\n⏰ {max_steps}ステップで終了")
    
    # 最終統計
    stats = agent.get_statistics()
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("📊 最終結果")
    print("="*70)
    
    print(f"\nゴール到達: {'✅' if agent.is_goal_reached() else '❌'}")
    print(f"総ステップ: {stats['steps']}")
    print(f"壁衝突率: {stats['wall_hit_rate']:.1%}")
    print(f"最終距離: {stats['distance_to_goal']}")
    print(f"エピソード数: {stats['episodes']}")
    print(f"グラフエッジ数: {stats['edges']}")
    print(f"平均geDIG: {stats['avg_gedig']:.3f}")
    print(f"実行時間: {elapsed:.1f}秒")
    
    # 効率性評価
    if agent.is_goal_reached():
        optimal_steps = 44  # マンハッタン距離
        efficiency = optimal_steps / stats['steps'] * 100
        print(f"\n効率性: {efficiency:.1f}% (最適経路比)")
    
    # DataStore保存
    agent.finalize()
    
    print("\n✨ 25×25迷路実験完了")


if __name__ == "__main__":
    test_25x25_maze()