#!/usr/bin/env python3
"""
真の完全迷路での実験
深さ優先探索(DFS)アルゴリズムで生成した完全迷路を使用
"""

import numpy as np
import random
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from pure_memory_agent_optimized import PureMemoryAgentOptimized


def generate_perfect_maze_dfs(size=(11, 11), seed=None):
    """
    深さ優先探索で完全迷路を生成
    - 全ての通路が繋がっている
    - ループなし
    - 任意の2点間の経路は唯一
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    height, width = size
    # 奇数サイズに調整
    if height % 2 == 0:
        height += 1
    if width % 2 == 0:
        width += 1
    
    # 初期化（全て壁）
    maze = np.ones((height, width), dtype=int)
    
    # スタート地点
    current = (1, 1)
    maze[current] = 0
    
    # スタック（バックトラック用）
    stack = [current]
    
    # 方向
    directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
    
    while stack:
        # 未訪問の隣接セルを探す
        neighbors = []
        y, x = current
        
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 < ny < height-1 and 0 < nx < width-1:
                if maze[ny, nx] == 1:  # 未訪問
                    neighbors.append((ny, nx, dy, dx))
        
        if neighbors:
            # ランダムに選択
            ny, nx, dy, dx = random.choice(neighbors)
            # 壁を削って通路を作る
            maze[y + dy//2, x + dx//2] = 0
            maze[ny, nx] = 0
            # 次のセルへ
            current = (ny, nx)
            stack.append(current)
        else:
            # バックトラック
            if stack:
                current = stack.pop()
    
    return maze


def analyze_perfect_maze(maze):
    """完全迷路の特性を分析"""
    height, width = maze.shape
    
    # グラフ構築
    graph = {}
    passages = []
    
    for i in range(height):
        for j in range(width):
            if maze[i, j] == 0:
                passages.append((i, j))
                neighbors = []
                for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ny, nx = i + dy, j + dx
                    if 0 <= ny < height and 0 <= nx < width and maze[ny, nx] == 0:
                        neighbors.append((ny, nx))
                graph[(i, j)] = neighbors
    
    # 特性計算
    junctions = sum(1 for node in graph if len(graph[node]) >= 3)
    dead_ends = sum(1 for node in graph if len(graph[node]) == 1)
    corridors = sum(1 for node in graph if len(graph[node]) == 2)
    
    # ループチェック（DFSで生成した場合は必ずFalse）
    def has_cycle():
        if not passages:
            return False
        
        visited = set()
        parent = {}
        
        def dfs(node, par):
            visited.add(node)
            parent[node] = par
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, node):
                        return True
                elif parent.get(node) != neighbor:
                    return True
            return False
        
        return dfs(passages[0], None)
    
    return {
        'passages': len(passages),
        'junctions': junctions,
        'dead_ends': dead_ends,
        'corridors': corridors,
        'has_loop': has_cycle(),
        'complexity': (junctions + dead_ends) / max(1, len(passages))
    }


def test_perfect_maze_dfs(size=(11, 11), seed=None):
    """DFS生成の完全迷路でテスト"""
    
    print("="*60)
    print(f"🎯 真の完全迷路テスト（{size[0]}×{size[1]}）")
    print("  DFSアルゴリズムで生成")
    print("="*60)
    
    # 迷路生成
    maze = generate_perfect_maze_dfs(size, seed)
    
    # 分析
    analysis = analyze_perfect_maze(maze)
    
    print("\n📊 迷路構造分析:")
    print(f"  完全迷路: {'✅' if not analysis['has_loop'] else '❌'}")
    print(f"  通路数: {analysis['passages']}マス")
    print(f"  分岐点: {analysis['junctions']}箇所")
    print(f"  袋小路: {analysis['dead_ends']}箇所")
    print(f"  廊下: {analysis['corridors']}マス")
    print(f"  複雑度: {analysis['complexity']:.2f}")
    
    print("\n迷路構造:")
    for i, row in enumerate(maze):
        row_str = ""
        for j, cell in enumerate(row):
            if i == 1 and j == 1:
                row_str += "S"
            elif i == size[0]-2 and j == size[1]-2:
                row_str += "G"
            elif cell == 1:
                row_str += "█"
            else:
                row_str += " "
        print(row_str)
    
    # エージェント作成
    agent = PureMemoryAgentOptimized(
        maze=maze,
        datastore_path="../results/true_perfect_maze",
        config={
            'max_depth': 5,
            'search_k': 30,
            'gedig_threshold': 0.5,
            'max_edges_per_node': 20
        }
    )
    
    initial_dist = abs(agent.position[0] - agent.goal[0]) + \
                   abs(agent.position[1] - agent.goal[1])
    
    print(f"\n📍 スタート: {agent.position}")
    print(f"🎯 ゴール: {agent.goal}")
    print(f"📏 マンハッタン距離: {initial_dist}")
    print("-" * 60)
    
    # 実行
    path = [agent.position]
    visited_junctions = set()
    visited_deadends = set()
    backtrack_count = 0
    
    print("\n実行中...")
    start_time = time.time()
    
    for step in range(200):  # 制限時間短縮
        if agent.is_goal_reached():
            elapsed = time.time() - start_time
            print(f"\n🎉 成功！ {step}ステップでゴール到達！")
            print(f"⏱️ 実行時間: {elapsed:.2f}秒")
            break
        
        # 現在位置の分析
        y, x = agent.position
        neighbors = sum([
            y > 0 and maze[y-1, x] == 0,
            y < maze.shape[0]-1 and maze[y+1, x] == 0,
            x > 0 and maze[y, x-1] == 0,
            x < maze.shape[1]-1 and maze[y, x+1] == 0
        ])
        
        if neighbors >= 3:
            visited_junctions.add((y, x))
        elif neighbors == 1 and (y, x) != agent.goal:
            visited_deadends.add((y, x))
        
        # バックトラック検出
        if len(path) >= 3 and agent.position in path[-3:-1]:
            backtrack_count += 1
        
        action = agent.get_action()
        agent.execute_action(action)
        path.append(agent.position)
        
        # 進捗
        if step % 50 == 49:
            stats = agent.get_statistics()
            print(f"\nStep {step+1}:")
            print(f"  位置: {agent.position}")
            print(f"  ゴールまでの距離: {stats['distance_to_goal']}")
            print(f"  分岐点探索: {len(visited_junctions)}/{analysis['junctions']}")
            print(f"  袋小路訪問: {len(visited_deadends)}/{analysis['dead_ends']}")
    else:
        print(f"\n⏰ {step+1}ステップで終了")
    
    # 経路表示
    print("\n📊 最終経路:")
    display_path_ascii(maze, path, agent.goal)
    
    # 結果分析
    final_stats = agent.get_statistics()
    
    print("\n" + "="*60)
    print("📊 結果分析")
    print("="*60)
    
    success = agent.is_goal_reached()
    print(f"\n基本結果:")
    print(f"  ゴール到達: {'✅ 成功' if success else '❌ 失敗'}")
    
    if success:
        print(f"  総ステップ: {final_stats['steps']}")
        print(f"  壁衝突率: {final_stats['wall_hit_rate']:.1%}")
        
        # 探索効率
        unique_positions = len(set(path))
        print(f"\n探索効率:")
        print(f"  訪問マス数: {unique_positions}")
        print(f"  総通路数: {analysis['passages']}")
        print(f"  探索カバー率: {unique_positions/analysis['passages']*100:.1f}%")
        
        # 分岐点と袋小路
        print(f"\n構造探索:")
        print(f"  分岐点発見: {len(visited_junctions)}/{analysis['junctions']}")
        print(f"  袋小路訪問: {len(visited_deadends)}/{analysis['dead_ends']}")
        print(f"  バックトラック回数: {backtrack_count}")
        
        # 学習品質
        print(f"\n学習品質:")
        print(f"  平均geDIG: {final_stats['avg_gedig']:.3f}")
        if final_stats['avg_gedig'] < 0:
            print(f"  → 良好（情報利得 > 編集距離）")
        
        # 深度使用
        print(f"\n推論深度:")
        total = sum(final_stats['depth_usage'].values())
        if total > 0:
            for depth in sorted(final_stats['depth_usage'].keys()):
                count = final_stats['depth_usage'][depth]
                if count > 0:
                    print(f"  {depth}ホップ: {count/total*100:.1f}%")


def display_path_ascii(maze, path, goal):
    """ASCII形式で経路表示"""
    height, width = maze.shape
    
    # 訪問順を記録
    visit_order = {}
    for i, pos in enumerate(path):
        if pos not in visit_order:
            visit_order[pos] = i
    
    for i in range(height):
        row_str = ""
        for j in range(width):
            pos = (i, j)
            
            if pos == path[0]:
                row_str += "S"
            elif pos == goal:
                row_str += "G"
            elif pos == path[-1] and pos != goal:
                row_str += "E"
            elif pos in visit_order:
                order = visit_order[pos]
                if order < 10:
                    row_str += str(order)
                elif order < 36:
                    row_str += chr(ord('A') + order - 10)
                elif order < 62:
                    row_str += chr(ord('a') + order - 36)
                else:
                    row_str += "*"
            elif maze[i, j] == 1:
                row_str += "█"
            else:
                row_str += " "
        print(row_str)
    
    print("\n凡例:")
    print("  S=スタート, G=ゴール, E=終了位置")
    print("  0-9,A-Z,a-z,*=訪問順")


if __name__ == "__main__":
    # 結果ディレクトリ作成
    os.makedirs('../results', exist_ok=True)
    
    print("🔬 真の完全迷路実験\n")
    
    # 11×11で3回（異なるシード）
    print("【11×11 完全迷路】")
    seeds = [42, 123, 789]
    
    for trial, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"試行 {trial + 1}/3 (seed={seed})")
        print('='*60)
        test_perfect_maze_dfs((11, 11), seed)
        if trial < 2:
            time.sleep(1)
    
    # 13×13で1回
    print("\n\n【13×13 より複雑な完全迷路】")
    test_perfect_maze_dfs((13, 13), seed=999)
    
    # 15×15で1回（チャレンジ）
    print("\n\n【15×15 高複雑度完全迷路（チャレンジ）】")
    test_perfect_maze_dfs((15, 15), seed=2024)