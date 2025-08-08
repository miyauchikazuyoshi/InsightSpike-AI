#!/usr/bin/env python3
"""
50×50迷路でのgeDIGテスト
topk=7, hop数最大20, タイムアウトなし
"""

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Tuple

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from pure_gedig_no_cheat import PureGedigNoCheat
from test_true_perfect_maze import generate_perfect_maze_dfs


def visualize_maze_progress(maze: np.ndarray, agent, step: int, filename: str):
    """迷路と進捗を可視化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 迷路全体
    ax1.imshow(maze, cmap='binary')
    ax1.plot(agent.position[1], agent.position[0], 'ro', markersize=8)
    ax1.plot(agent.goal[1], agent.goal[0], 'g*', markersize=12)
    ax1.set_title(f"50×50 Maze - Step {step}")
    ax1.axis('off')
    
    # ズームビュー（現在位置周辺）
    x, y = agent.position
    window = 10
    x_min = max(0, x - window)
    x_max = min(maze.shape[0], x + window + 1)
    y_min = max(0, y - window)
    y_max = min(maze.shape[1], y + window + 1)
    
    zoomed = maze[x_min:x_max, y_min:y_max]
    ax2.imshow(zoomed, cmap='binary')
    
    # ズーム内での相対位置
    rel_x = x - x_min
    rel_y = y - y_min
    ax2.plot(rel_y, rel_x, 'ro', markersize=12)
    
    # ゴールがズーム内にあれば表示
    if x_min <= agent.goal[0] < x_max and y_min <= agent.goal[1] < y_max:
        goal_rel_x = agent.goal[0] - x_min
        goal_rel_y = agent.goal[1] - y_min
        ax2.plot(goal_rel_y, goal_rel_x, 'g*', markersize=15)
    
    ax2.set_title(f"Zoomed View (±{window} cells)")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()


def test_50x50_maze():
    """50×50迷路でのテスト"""
    print("="*70)
    print("🌟 50×50迷路 geDIG実装テスト")
    print("="*70)
    
    # 50×50迷路生成
    print("\n🔨 50×50迷路を生成中...")
    maze = generate_perfect_maze_dfs((50, 50), seed=42)
    
    print(f"  迷路サイズ: {maze.shape}")
    print(f"  スタート: (1, 1)")
    print(f"  ゴール: (48, 48)")
    print(f"  最短距離（マンハッタン）: {47 + 47} = 94")
    
    # エージェント作成
    print("\n🤖 エージェント初期化...")
    agent = PureGedigNoCheat(
        maze=maze,
        datastore_path="data/maze_50x50_gedig",
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
    checkpoint_steps = [100, 500, 1000, 2500, 5000, 10000]
    checkpoint_idx = 0
    
    max_steps = 20000  # 十分な上限
    
    for step in range(max_steps):
        if agent.is_goal_reached():
            elapsed = time.time() - start_time
            print(f"\n🎉 成功！")
            print(f"  ステップ数: {step}")
            print(f"  実行時間: {elapsed:.1f}秒")
            
            # 最終ビジュアライゼーション
            visualize_maze_progress(
                maze, agent, step,
                "../results/50x50_maze_success.png"
            )
            break
        
        # 行動決定と実行
        action = agent.get_action()
        success = agent.execute_action(action)
        
        # チェックポイント
        if checkpoint_idx < len(checkpoint_steps) and step + 1 == checkpoint_steps[checkpoint_idx]:
            stats = agent.get_statistics()
            elapsed = time.time() - start_time
            
            print(f"\n📊 チェックポイント - Step {step + 1}:")
            print(f"  位置: {agent.position}")
            print(f"  ゴールまでの距離: {stats['distance_to_goal']}")
            print(f"  壁衝突率: {stats['wall_hit_rate']:.1%}")
            print(f"  エピソード数: {stats['episodes']}")
            print(f"  グラフエッジ数: {stats['edges']}")
            print(f"  平均geDIG: {stats['avg_gedig']:.3f}")
            print(f"  経過時間: {elapsed:.1f}秒")
            
            # ビジュアライゼーション
            visualize_maze_progress(
                maze, agent, step + 1,
                f"../results/50x50_maze_step_{step + 1}.png"
            )
            
            checkpoint_idx += 1
        
        # 定期進捗（詳細）
        if step > 0 and step % 100 == 0:
            stats = agent.get_statistics()
            print(f"  Step {step}: 位置{agent.position}, 距離{stats['distance_to_goal']}, エピソード{stats['episodes']}, エッジ{stats['edges']}")
    
    else:
        print(f"\n⏰ {max_steps}ステップで終了")
        visualize_maze_progress(
            maze, agent, max_steps,
            "../results/50x50_maze_timeout.png"
        )
    
    # 最終統計
    stats = agent.get_statistics()
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("📊 最終結果")
    print("="*70)
    
    print(f"\nゴール到達: {'✅' if agent.is_goal_reached() else '❌'}")
    print(f"総ステップ: {stats['steps']}")
    print(f"壁衝突: {stats['wall_hits']}")
    print(f"壁衝突率: {stats['wall_hit_rate']:.1%}")
    print(f"最終距離: {stats['distance_to_goal']}")
    print(f"エピソード数: {stats['episodes']}")
    print(f"グラフエッジ数: {stats['edges']}")
    print(f"平均geDIG: {stats['avg_gedig']:.3f}")
    print(f"実行時間: {elapsed:.1f}秒")
    
    # 効率性評価
    if agent.is_goal_reached():
        optimal_steps = 94  # マンハッタン距離
        efficiency = optimal_steps / stats['steps'] * 100
        print(f"\n効率性: {efficiency:.1f}% (最適経路比)")
    
    # DataStore保存
    agent.finalize()
    
    print("\n✨ 50×50迷路実験完了")


if __name__ == "__main__":
    test_50x50_maze()