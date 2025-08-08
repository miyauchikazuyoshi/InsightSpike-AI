#!/usr/bin/env python3
"""
より複雑な迷路でのテスト
複数の経路、袋小路、ループを含む迷路で実験
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent_optimized import PureMemoryAgentOptimized


def create_complex_maze(size=(15, 15), complexity=0.8, density=0.8):
    """
    より複雑な迷路を生成
    complexity: 分岐の複雑さ (0.0-1.0)
    density: 壁の密度 (0.0-1.0)
    """
    height, width = size
    maze = np.zeros((height, width), dtype=int)
    
    # 外壁
    maze[0, :] = maze[-1, :] = 1
    maze[:, 0] = maze[:, -1] = 1
    
    # ランダムな壁を追加（複雑性を増す）
    complexity = int(complexity * (5 * (height + width)))
    density = int(density * ((height // 2) * (width // 2)))
    
    # 壁の位置をランダムに決定
    for _ in range(density):
        x = np.random.randint(0, width // 2) * 2
        y = np.random.randint(0, height // 2) * 2
        
        if x == 0 or y == 0 or x == width - 1 or y == height - 1:
            continue
            
        maze[y, x] = 1
        
        # 複雑な分岐を作成
        for _ in range(complexity):
            neighbours = []
            if x > 1:
                neighbours.append((y, x - 2))
            if x < width - 2:
                neighbours.append((y, x + 2))
            if y > 1:
                neighbours.append((y - 2, x))
            if y < height - 2:
                neighbours.append((y + 2, x))
            
            if neighbours:
                y_, x_ = neighbours[np.random.randint(0, len(neighbours))]
                
                if maze[y_, x_] == 0:
                    maze[y_, x_] = 1
                    maze[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_
    
    # スタートとゴールを確保
    maze[1, 1] = 0
    maze[-2, -2] = 0
    
    # パスを保証（最低限の通路を確保）
    ensure_path(maze)
    
    return maze


def ensure_path(maze):
    """最低限の通路を確保して解ける迷路にする"""
    height, width = maze.shape
    
    # 簡単な経路を1本確保
    # 横に進む
    for j in range(1, width // 2):
        maze[1, j] = 0
    
    # 縦に進む
    for i in range(1, height - 1):
        maze[i, width // 2] = 0
    
    # ゴールへの経路
    for j in range(width // 2, width - 1):
        maze[height - 2, j] = 0
    
    # いくつかの代替経路を追加
    for _ in range(3):
        start_y = np.random.randint(2, height - 2)
        start_x = np.random.randint(2, width - 2)
        
        # ランダムウォークで経路を作成
        y, x = start_y, start_x
        for _ in range(10):
            maze[y, x] = 0
            direction = np.random.choice(['up', 'down', 'left', 'right'])
            
            if direction == 'up' and y > 1:
                y -= 1
            elif direction == 'down' and y < height - 2:
                y += 1
            elif direction == 'left' and x > 1:
                x -= 1
            elif direction == 'right' and x < width - 2:
                x += 1


def test_complex_maze():
    """複雑な迷路でテスト"""
    
    print("="*70)
    print("🌀 複雑な迷路でのテスト")
    print("  複数経路、袋小路、ループを含む迷路")
    print("="*70)
    
    # 15×15の複雑な迷路を生成
    maze = create_complex_maze(size=(15, 15), complexity=0.75, density=0.75)
    
    print("\n生成された迷路 (15×15):")
    for row in maze:
        print(''.join(['█' if x == 1 else ' ' for x in row]))
    
    # 迷路の複雑さを分析
    analyze_maze_complexity(maze)
    
    # エージェント作成（より高度な設定）
    agent = PureMemoryAgentOptimized(
        maze=maze,
        datastore_path="../results/complex_maze",
        config={
            'max_depth': 7,      # より深い推論
            'search_k': 50,      # より多くの候補
            'gedig_threshold': 0.4,  # より厳密な評価
            'max_edges_per_node': 25  # より豊富なグラフ
        }
    )
    
    initial_distance = abs(agent.position[0] - agent.goal[0]) + \
                      abs(agent.position[1] - agent.goal[1])
    
    print(f"\n📍 スタート: {agent.position}")
    print(f"🎯 ゴール: {agent.goal}")
    print(f"📏 初期マンハッタン距離: {initial_distance}")
    print("-" * 70)
    
    # 実行と記録
    path = [agent.position]
    wall_hits = 0
    backtrack_count = 0
    previous_positions = []
    
    print("\n実行中...")
    start_time = time.time()
    
    for step in range(500):  # より多くのステップを許可
        if agent.is_goal_reached():
            elapsed = time.time() - start_time
            print(f"\n🎉 成功！ {step}ステップでゴール到達！")
            print(f"⏱️ 実行時間: {elapsed:.2f}秒")
            break
        
        # バックトラック検出
        if len(previous_positions) >= 5:
            if agent.position in previous_positions[-5:]:
                backtrack_count += 1
        
        previous_positions.append(agent.position)
        
        # 行動実行
        action = agent.get_action()
        success = agent.execute_action(action)
        
        if not success:
            wall_hits += 1
        
        path.append(agent.position)
        
        # 進捗報告
        if step % 50 == 49:
            stats = agent.get_statistics()
            print(f"\nStep {step+1}:")
            print(f"  現在位置: {agent.position}")
            print(f"  距離: {stats['distance_to_goal']}")
            print(f"  壁衝突率: {stats['wall_hit_rate']:.1%}")
            print(f"  バックトラック: {backtrack_count}回")
            
            # 深度使用
            total_depth = sum(stats['depth_usage'].values())
            if total_depth > 0:
                deep = sum(stats['depth_usage'].get(d, 0) for d in range(5, 8))
                print(f"  深い推論(5-7ホップ): {deep/total_depth*100:.1f}%")
    else:
        print(f"\n⏰ タイムアウト ({step+1}ステップ)")
    
    # 結果の可視化
    visualize_complex_maze_result(maze, path, agent.goal)
    
    # 最終統計
    final_stats = agent.get_statistics()
    
    print("\n" + "="*70)
    print("📊 最終結果")
    print("="*70)
    
    print(f"\n基本統計:")
    print(f"  ゴール到達: {'✅ 成功' if agent.is_goal_reached() else '❌ 未到達'}")
    print(f"  総ステップ: {final_stats['steps']}")
    print(f"  壁衝突回数: {wall_hits}")
    print(f"  壁衝突率: {final_stats['wall_hit_rate']:.1%}")
    print(f"  バックトラック: {backtrack_count}回")
    
    print(f"\n経路分析:")
    unique_positions = len(set(path))
    print(f"  訪問位置数: {unique_positions}")
    print(f"  再訪問率: {(len(path) - unique_positions) / len(path) * 100:.1f}%")
    
    print(f"\n学習品質:")
    print(f"  平均geDIG: {final_stats['avg_gedig']:.3f}")
    if final_stats['avg_gedig'] < 0:
        print(f"  → 良好な学習（情報利得 > 編集距離）")
    
    print(f"\nグラフ構造:")
    print(f"  ノード数: {final_stats['graph_nodes']}")
    print(f"  エッジ数: {final_stats['graph_edges']}")
    
    # 効率性
    if agent.is_goal_reached():
        optimal = initial_distance
        actual = len(path) - 1
        efficiency = optimal / actual * 100
        print(f"\n効率性:")
        print(f"  最適経路: {optimal}マス")
        print(f"  実際の経路: {actual}ステップ")
        print(f"  経路効率: {efficiency:.1f}%")


def analyze_maze_complexity(maze):
    """迷路の複雑さを分析"""
    height, width = maze.shape
    
    # 通路と壁の比率
    passages = np.sum(maze == 0)
    walls = np.sum(maze == 1)
    
    # 分岐点を数える（3方向以上に進める点）
    junctions = 0
    dead_ends = 0
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            if maze[i, j] == 0:  # 通路の場合
                # 隣接する通路を数える
                neighbors = 0
                if maze[i-1, j] == 0:
                    neighbors += 1
                if maze[i+1, j] == 0:
                    neighbors += 1
                if maze[i, j-1] == 0:
                    neighbors += 1
                if maze[i, j+1] == 0:
                    neighbors += 1
                
                if neighbors >= 3:
                    junctions += 1
                elif neighbors == 1:
                    dead_ends += 1
    
    print(f"\n📊 迷路の複雑さ分析:")
    print(f"  サイズ: {height}×{width}")
    print(f"  通路: {passages}マス ({passages/(height*width)*100:.1f}%)")
    print(f"  壁: {walls}マス ({walls/(height*width)*100:.1f}%)")
    print(f"  分岐点: {junctions}箇所")
    print(f"  袋小路: {dead_ends}箇所")
    print(f"  複雑度スコア: {(junctions + dead_ends) / passages * 100:.1f}")


def visualize_complex_maze_result(maze, path, goal):
    """複雑な迷路の結果を可視化"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. 迷路と完全な経路
    ax = axes[0]
    plot_maze_with_path(ax, maze, path, goal, "Complete Path")
    
    # 2. 訪問頻度ヒートマップ
    ax = axes[1]
    plot_visit_heatmap(ax, maze, path, "Visit Frequency")
    
    # 3. フェーズごとの進行
    ax = axes[2]
    plot_phase_progression(ax, maze, path, goal, "Phase Progression")
    
    plt.suptitle('Complex Maze Navigation Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/complex_maze_result.png', dpi=150, bbox_inches='tight')
    print("\n✅ 可視化を保存: results/complex_maze_result.png")


def plot_maze_with_path(ax, maze, path, goal, title):
    """迷路と経路を描画"""
    height, width = maze.shape
    
    # 迷路を描画
    for i in range(height):
        for j in range(width):
            if maze[i, j] == 1:
                rect = patches.Rectangle((j-0.5, i-0.5), 1, 1,
                                        linewidth=0, facecolor='black')
                ax.add_patch(rect)
    
    # 経路を描画
    if path:
        for i in range(len(path)-1):
            color = plt.cm.viridis(i / len(path))
            ax.plot([path[i][1], path[i+1][1]], 
                   [path[i][0], path[i+1][0]], 
                   color=color, alpha=0.7, linewidth=2)
        
        # スタートとゴール
        ax.plot(path[0][1], path[0][0], 'go', markersize=12, label='Start')
        ax.plot(goal[1], goal[0], 'r*', markersize=15, label='Goal')
        if path[-1] != goal:
            ax.plot(path[-1][1], path[-1][0], 'bo', markersize=10, label='Final')
    
    ax.set_xlim(-0.5, width-0.5)
    ax.set_ylim(-0.5, height-0.5)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)


def plot_visit_heatmap(ax, maze, path, title):
    """訪問頻度のヒートマップ"""
    height, width = maze.shape
    visit_count = np.zeros((height, width))
    
    for pos in path:
        visit_count[pos[0], pos[1]] += 1
    
    # マスクを作成（壁の部分）
    masked_visits = np.ma.masked_where(maze == 1, visit_count)
    
    im = ax.imshow(masked_visits, cmap='hot', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Visit Count')
    
    ax.set_title(title)
    ax.set_xlim(-0.5, width-0.5)
    ax.set_ylim(-0.5, height-0.5)


def plot_phase_progression(ax, maze, path, goal, title):
    """フェーズごとの進行を可視化"""
    height, width = maze.shape
    
    # 背景の迷路
    for i in range(height):
        for j in range(width):
            if maze[i, j] == 1:
                rect = patches.Rectangle((j-0.5, i-0.5), 1, 1,
                                        linewidth=0, facecolor='lightgray')
                ax.add_patch(rect)
    
    # フェーズごとに色分け
    phases = [
        (0, len(path)//3, 'blue', 'Early'),
        (len(path)//3, 2*len(path)//3, 'purple', 'Middle'),
        (2*len(path)//3, len(path), 'red', 'Late')
    ]
    
    for start, end, color, label in phases:
        if start < len(path):
            segment = path[start:min(end, len(path))]
            for i in range(len(segment)-1):
                ax.plot([segment[i][1], segment[i+1][1]], 
                       [segment[i][0], segment[i+1][0]], 
                       color=color, alpha=0.6, linewidth=2, label=label if i==0 else '')
    
    # マーカー
    if path:
        ax.plot(path[0][1], path[0][0], 'go', markersize=12)
        ax.plot(goal[1], goal[0], 'r*', markersize=15)
    
    ax.set_xlim(-0.5, width-0.5)
    ax.set_ylim(-0.5, height-0.5)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)


if __name__ == "__main__":
    # 結果ディレクトリ作成
    os.makedirs('../results', exist_ok=True)
    
    # テスト実行
    test_complex_maze()