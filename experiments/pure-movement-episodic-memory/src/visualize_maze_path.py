#!/usr/bin/env python3
"""
実際の迷路と経路の詳細可視化
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from insightspike.environments.proper_maze_generator import ProperMazeGenerator
from pure_memory_agent_optimized import PureMemoryAgentOptimized


def visualize_maze_and_path():
    """迷路と実際の経路を可視化"""
    
    print("🗺️ 実際の迷路と経路を可視化中...")
    
    # 11×11迷路生成（実験と同じシード）
    generator = ProperMazeGenerator()
    maze = generator.generate_dfs_maze(size=(11, 11), seed=789)
    
    # エージェント作成
    agent = PureMemoryAgentOptimized(
        maze=maze,
        datastore_path="../results/path_visualization",
        config={
            'max_depth': 5,
            'search_k': 30,
            'gedig_threshold': 0.5
        }
    )
    
    # 経路を記録しながら実行
    path = [agent.position]
    wall_hits = []
    
    print("\n実行中...")
    for step in range(200):
        if agent.is_goal_reached():
            print(f"✅ {step}ステップでゴール到達！")
            break
        
        prev_pos = agent.position
        action = agent.get_action()
        success = agent.execute_action(action)
        
        if not success:
            # 壁衝突位置を記録
            dx, dy = agent.action_deltas[action]
            wall_hit_pos = (prev_pos[0] + dx, prev_pos[1] + dy)
            wall_hits.append((prev_pos, wall_hit_pos, step))
        
        path.append(agent.position)
    
    # Figure作成（高解像度）
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 元の迷路
    ax = axes[0, 0]
    plot_maze(ax, maze, title="1. Original Maze (11×11)")
    
    # スタートとゴールをマーク
    start = path[0]
    goal = agent.goal
    ax.plot(start[1], start[0], 'go', markersize=15, label='Start')
    ax.plot(goal[1], goal[0], 'r*', markersize=20, label='Goal')
    ax.legend()
    
    # 2. 完全な経路
    ax = axes[0, 1]
    plot_maze(ax, maze, title="2. Complete Path")
    plot_full_path(ax, path, start, goal)
    
    # 3. 経路のヒートマップ
    ax = axes[0, 2]
    plot_maze(ax, maze, title="3. Visit Frequency Heatmap")
    plot_heatmap(ax, path, maze.shape)
    
    # 4. 初期探索フェーズ（最初の30ステップ）
    ax = axes[1, 0]
    plot_maze(ax, maze, title="4. Initial Exploration (Steps 1-30)")
    if len(path) > 30:
        plot_path_segment(ax, path[:31], start, color='blue', alpha_range=(0.3, 0.8))
    
    # 5. 中盤フェーズ（31-60ステップ）
    ax = axes[1, 1]
    plot_maze(ax, maze, title="5. Middle Phase (Steps 31-60)")
    if len(path) > 60:
        plot_path_segment(ax, path[30:61], None, color='purple', alpha_range=(0.3, 0.8))
    
    # 6. 最終フェーズ（61ステップ以降）
    ax = axes[1, 2]
    plot_maze(ax, maze, title="6. Final Phase (Steps 61+)")
    if len(path) > 60:
        plot_path_segment(ax, path[60:], goal, color='red', alpha_range=(0.3, 0.8))
        ax.plot(goal[1], goal[0], 'r*', markersize=20)
    
    # 全体タイトル
    fig.suptitle('Maze Navigation with OptimizedNumpyIndex - Detailed Path Analysis', 
                fontsize=16, fontweight='bold')
    
    # 統計情報を追加
    stats_text = f"""
    Total Steps: {len(path)-1}
    Wall Hits: {len(wall_hits)}
    Success Rate: {(1 - len(wall_hits)/max(1, len(path)-1))*100:.1f}%
    Start: {start}
    Goal: {goal}
    Manhattan Distance: {abs(start[0]-goal[0]) + abs(start[1]-goal[1])}
    """
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/maze_path_visualization.png', dpi=150, bbox_inches='tight')
    print("✅ 保存: results/maze_path_visualization.png")
    
    # ASCII形式でも表示
    print("\n📊 ASCII形式の迷路と最終経路:")
    print_ascii_maze_with_path(maze, path, agent.goal)
    
    return maze, path, wall_hits


def plot_maze(ax, maze, title=""):
    """迷路の基本描画"""
    height, width = maze.shape
    
    # グリッド設定
    ax.set_xlim(-0.5, width-0.5)
    ax.set_ylim(-0.5, height-0.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.grid(True, alpha=0.2)
    
    # 壁を描画
    for i in range(height):
        for j in range(width):
            if maze[i, j] == 1:
                rect = patches.Rectangle((j-0.5, i-0.5), 1, 1,
                                        linewidth=0, facecolor='black')
                ax.add_patch(rect)


def plot_full_path(ax, path, start, goal):
    """完全な経路を描画"""
    # スタートとゴール
    ax.plot(start[1], start[0], 'go', markersize=15, label='Start', zorder=5)
    ax.plot(goal[1], goal[0], 'r*', markersize=20, label='Goal', zorder=5)
    
    # 経路を線で描画（グラデーション）
    for i in range(len(path)-1):
        alpha = 0.3 + 0.7 * i / len(path)
        color = plt.cm.viridis(i / len(path))
        ax.plot([path[i][1], path[i+1][1]], 
               [path[i][0], path[i+1][0]], 
               color=color, alpha=alpha, linewidth=2)
    
    # 現在位置
    if path:
        ax.plot(path[-1][1], path[-1][0], 'bo', markersize=10, label='Final Position')
    
    ax.legend(loc='upper right', fontsize=10)


def plot_heatmap(ax, path, shape):
    """訪問頻度のヒートマップ"""
    height, width = shape
    visit_count = np.zeros((height, width))
    
    for pos in path:
        visit_count[pos[0], pos[1]] += 1
    
    # ヒートマップ描画
    im = ax.imshow(visit_count, cmap='hot', interpolation='nearest', alpha=0.8)
    plt.colorbar(im, ax=ax, label='Visit Count')
    
    # スタートとゴールをマーク
    if path:
        ax.plot(path[0][1], path[0][0], 'go', markersize=15)
        ax.plot(path[-1][1], path[-1][0], 'bo', markersize=10)


def plot_path_segment(ax, segment, marker_pos, color='blue', alpha_range=(0.3, 0.8)):
    """経路の一部を描画"""
    for i in range(len(segment)-1):
        alpha = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * i / len(segment)
        ax.plot([segment[i][1], segment[i+1][1]], 
               [segment[i][0], segment[i+1][0]], 
               color=color, alpha=alpha, linewidth=2)
    
    # 開始点と終了点
    if segment:
        ax.plot(segment[0][1], segment[0][0], 'o', color=color, markersize=8, alpha=0.8)
        ax.plot(segment[-1][1], segment[-1][0], 's', color=color, markersize=8)
    
    # 特別なマーカー
    if marker_pos:
        ax.plot(marker_pos[1], marker_pos[0], 'g*', markersize=15)


def print_ascii_maze_with_path(maze, path, goal):
    """ASCII形式で迷路と経路を表示"""
    height, width = maze.shape
    
    # 経路を辞書に変換（位置→ステップ番号）
    path_dict = {}
    for i, pos in enumerate(path):
        if pos not in path_dict:  # 最初の訪問のみ記録
            path_dict[pos] = i
    
    # 各セルの文字を決定
    for i in range(height):
        row_str = ""
        for j in range(width):
            pos = (i, j)
            
            if pos == path[0]:
                row_str += "S"  # スタート
            elif pos == goal:
                row_str += "G"  # ゴール
            elif pos == path[-1] and pos != goal:
                row_str += "E"  # 終了位置（ゴール未到達の場合）
            elif pos in path_dict:
                # 訪問順を数字で表示（0-9, A-Z, その後は*）
                step = path_dict[pos]
                if step < 10:
                    row_str += str(step)
                elif step < 36:
                    row_str += chr(ord('A') + step - 10)
                else:
                    row_str += "*"
            elif maze[i, j] == 1:
                row_str += "█"  # 壁
            else:
                row_str += " "  # 未訪問の通路
        
        print(row_str)
    
    print("\n凡例: S=スタート, G=ゴール, 0-9,A-Z=訪問順, █=壁")


if __name__ == "__main__":
    # 結果ディレクトリ作成
    os.makedirs('../results', exist_ok=True)
    
    # 実行
    maze, path, wall_hits = visualize_maze_and_path()
    
    print(f"\n📊 統計:")
    print(f"  総ステップ数: {len(path)-1}")
    print(f"  壁衝突回数: {len(wall_hits)}")
    print(f"  ユニークな訪問位置: {len(set(path))}")
    
    # 経路の効率性分析
    start = path[0]
    goal = path[-1]
    optimal_distance = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    actual_distance = len(path) - 1
    
    if optimal_distance > 0:
        efficiency = optimal_distance / actual_distance * 100
        print(f"\n  最適経路長: {optimal_distance}")
        print(f"  実際の経路長: {actual_distance}")
        print(f"  経路効率: {efficiency:.1f}%")