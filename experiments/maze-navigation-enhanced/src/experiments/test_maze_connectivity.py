#!/usr/bin/env python3
"""
迷路の接続性をチェック
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import deque

def create_complex_maze():
    """複雑な迷路（多数の分岐、袋小路、ループ構造）"""
    maze = np.ones((21, 21), dtype=int)
    
    # メインの縦通路（複数）
    for y in range(1, 20):
        maze[y, 3] = 0   # 左側縦通路
        maze[y, 10] = 0  # 中央縦通路
        maze[y, 17] = 0  # 右側縦通路
    
    # 横の接続通路（複数レベル）
    for x in range(1, 20):
        maze[5, x] = 0   # 上部横通路
        maze[10, x] = 0  # 中央横通路
        maze[15, x] = 0  # 下部横通路
    
    # 袋小路を追加（様々な深さ）
    
    # 短い袋小路（2-3マス）
    maze[3, 5] = 0
    maze[3, 6] = 0
    
    maze[7, 18] = 0
    maze[7, 19] = 0
    
    # 中程度の袋小路（4-5マス）
    for x in range(5, 10):
        maze[8, x] = 0
    maze[9, 8] = 1  # 壁を戻して袋小路に
    
    for y in range(12, 16):
        maze[y, 5] = 0
    maze[13, 4] = 0
    maze[14, 6] = 0
    
    # 長い袋小路（6マス以上）
    for x in range(11, 17):
        maze[13, x] = 0
    maze[14, 14] = 0
    maze[12, 13] = 0
    maze[13, 10] = 1  # 壁を戻して袋小路に
    
    # 迷わせる構造（ループ）
    for y in range(17, 20):
        maze[y, 7] = 0
        maze[y, 13] = 0
    maze[18, 8] = 0
    maze[18, 9] = 0
    maze[18, 11] = 0
    maze[18, 12] = 0
    
    # いくつかの通路を塞いで難易度調整
    maze[10, 3] = 1   # 左側を部分的に塞ぐ
    maze[10, 17] = 1  # 右側を部分的に塞ぐ
    maze[5, 10] = 1   # 中央上部を塞ぐ
    
    return maze


def check_connectivity(maze, start, goal):
    """BFSで接続性をチェック"""
    h, w = maze.shape
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        x, y = queue.popleft()
        
        if (x, y) == goal:
            return True, visited
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and maze[ny, nx] == 0:
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    
    return False, visited


def visualize_maze_connectivity(maze, start_pos, goal_pos):
    """迷路と接続性を可視化"""
    h, w = maze.shape
    
    # 接続性チェック
    is_connected, reachable = check_connectivity(maze, start_pos, goal_pos)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左: 迷路構造
    for y in range(h):
        for x in range(w):
            if maze[y, x] == 1:
                rect = Rectangle((x-0.5, y-0.5), 1, 1, 
                               facecolor='black', edgecolor='gray')
                ax1.add_patch(rect)
    
    ax1.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
    ax1.plot(goal_pos[0], goal_pos[1], 'ro', markersize=10, label='Goal')
    
    ax1.set_xlim(-0.5, w-0.5)
    ax1.set_ylim(-0.5, h-0.5)
    ax1.invert_yaxis()
    ax1.set_aspect('equal')
    ax1.set_title(f'Maze Structure\nConnected: {is_connected}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右: 到達可能エリア
    for y in range(h):
        for x in range(w):
            if maze[y, x] == 1:
                rect = Rectangle((x-0.5, y-0.5), 1, 1, 
                               facecolor='black', edgecolor='gray')
                ax2.add_patch(rect)
            elif (x, y) in reachable:
                rect = Rectangle((x-0.5, y-0.5), 1, 1, 
                               facecolor='lightgreen', edgecolor='green', alpha=0.5)
                ax2.add_patch(rect)
    
    ax2.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
    ax2.plot(goal_pos[0], goal_pos[1], 'ro', markersize=10, label='Goal')
    
    ax2.set_xlim(-0.5, w-0.5)
    ax2.set_ylim(-0.5, h-0.5)
    ax2.invert_yaxis()
    ax2.set_aspect('equal')
    ax2.set_title(f'Reachable Area from Start\n({len(reachable)} cells)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Maze Connectivity Check', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig, is_connected, reachable


def create_better_complex_maze():
    """改善された複雑な迷路（確実に接続されている）"""
    maze = np.ones((21, 21), dtype=int)
    
    # メインの通路網
    # 縦の基幹通路
    for y in range(1, 20):
        maze[y, 10] = 0  # 中央縦通路
    
    # 横の基幹通路
    for x in range(1, 20):
        maze[5, x] = 0   # 上部横通路
        maze[10, x] = 0  # 中央横通路
        maze[15, x] = 0  # 下部横通路
    
    # 左右の縦通路（部分的）
    for y in range(1, 10):
        maze[y, 3] = 0   # 左側上部
    for y in range(11, 20):
        maze[y, 3] = 0   # 左側下部
    
    for y in range(1, 9):
        maze[y, 17] = 0  # 右側上部
    for y in range(12, 20):
        maze[y, 17] = 0  # 右側下部
    
    # 袋小路を追加（接続を壊さないように）
    
    # 短い袋小路
    maze[3, 5] = 0
    maze[3, 6] = 0
    
    maze[7, 18] = 0
    maze[7, 19] = 0
    
    # 中程度の袋小路
    for x in range(5, 8):
        maze[8, x] = 0
    
    for y in range(12, 14):
        maze[y, 5] = 0
    
    # 長い袋小路
    for x in range(12, 16):
        maze[13, x] = 0
    maze[14, 14] = 0
    
    # 追加の接続路（迷路を面白くする）
    maze[7, 7] = 0
    maze[7, 8] = 0
    maze[13, 7] = 0
    maze[13, 8] = 0
    
    return maze


def main():
    print("="*60)
    print("MAZE CONNECTIVITY CHECK")
    print("="*60)
    
    # 元の迷路をチェック
    maze1 = create_complex_maze()
    start_pos = (3, 19)
    goal_pos = (17, 1)
    
    print("\n元の迷路:")
    print(f"Start: {start_pos}, Goal: {goal_pos}")
    
    fig1, connected1, reachable1 = visualize_maze_connectivity(maze1, start_pos, goal_pos)
    
    print(f"接続性: {connected1}")
    print(f"到達可能セル数: {len(reachable1)}")
    
    if not connected1:
        print("\n⚠️ 問題: スタートからゴールに到達できません！")
        
        # ゴールが到達可能エリアに含まれているか確認
        if goal_pos not in reachable1:
            print(f"ゴール {goal_pos} は到達可能エリア外です")
    
    # 改善された迷路をチェック
    print("\n" + "="*60)
    print("改善された迷路:")
    
    maze2 = create_better_complex_maze()
    fig2, connected2, reachable2 = visualize_maze_connectivity(maze2, start_pos, goal_pos)
    
    print(f"接続性: {connected2}")
    print(f"到達可能セル数: {len(reachable2)}")
    
    if connected2:
        print("✅ スタートからゴールに到達可能です！")
    
    # 迷路の詳細表示
    print("\n改善された迷路の構造:")
    h, w = maze2.shape
    for y in range(h):
        row = ""
        for x in range(w):
            if (x, y) == start_pos:
                row += "S "
            elif (x, y) == goal_pos:
                row += "G "
            elif maze2[y, x] == 1:
                row += "█ "
            else:
                if (x, y) in reachable2:
                    row += "· "
                else:
                    row += "? "  # 到達不可能な通路
        print(row)
    
    plt.show()


if __name__ == "__main__":
    main()