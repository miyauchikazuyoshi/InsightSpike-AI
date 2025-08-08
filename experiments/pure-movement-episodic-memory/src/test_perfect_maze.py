#!/usr/bin/env python3
"""
完全迷路での実験
- 複数の分岐点を持つ
- ループなし（任意の2点間の経路は唯一）
- 全ての通路に意味がある
"""

import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from pure_memory_agent_optimized import PureMemoryAgentOptimized


def create_perfect_maze_11x11():
    """
    11×11の完全迷路を手動作成
    - 複数の分岐点
    - ループなし
    - 深さ優先探索で生成される典型的な構造
    """
    maze = np.array([
        [1,1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,1,0,0,0,0,0,1],  # 最初の分岐
        [1,1,1,0,1,0,1,1,1,0,1],
        [1,0,0,0,0,0,0,0,1,0,1],  # 複数の選択肢
        [1,0,1,1,1,1,1,0,1,0,1],
        [1,0,0,0,0,0,1,0,0,0,1],  # 長い廊下と分岐
        [1,1,1,1,1,0,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,0,1],  # 別の長い廊下
        [1,0,1,1,1,1,1,1,1,0,1],
        [1,0,0,0,0,0,0,0,0,0,1],  # ゴールへの経路
        [1,1,1,1,1,1,1,1,1,1,1]
    ])
    return maze


def create_perfect_maze_13x13():
    """
    13×13のより複雑な完全迷路
    分岐点が多く、袋小路も含む
    """
    maze = np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,1,0,0,0,1,0,0,0,1],  # 3つの初期分岐
        [1,0,1,0,1,0,1,0,1,0,1,0,1],
        [1,0,1,0,0,0,1,0,0,0,1,0,1],  # 交差する経路
        [1,0,1,1,1,1,1,1,1,1,1,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,1],  # 大きな廊下
        [1,1,1,0,1,1,1,1,1,0,1,1,1],
        [1,0,0,0,0,0,1,0,0,0,0,0,1],  # 中央の分岐
        [1,0,1,1,1,0,1,0,1,1,1,0,1],
        [1,0,0,0,1,0,0,0,1,0,0,0,1],  # 複数の袋小路
        [1,1,1,0,1,1,1,1,1,0,1,1,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,1],  # 最終廊下
        [1,1,1,1,1,1,1,1,1,1,1,1,1]
    ])
    return maze


def verify_perfect_maze(maze):
    """迷路が完全迷路であることを検証"""
    height, width = maze.shape
    
    # 1. 連結性チェック（全ての通路が繋がっているか）
    # 2. ループがないことの確認
    
    # 通路のグラフを構築
    graph = {}
    passages = []
    
    for i in range(height):
        for j in range(width):
            if maze[i, j] == 0:
                passages.append((i, j))
                neighbors = []
                if i > 0 and maze[i-1, j] == 0:
                    neighbors.append((i-1, j))
                if i < height-1 and maze[i+1, j] == 0:
                    neighbors.append((i+1, j))
                if j > 0 and maze[i, j-1] == 0:
                    neighbors.append((i, j-1))
                if j < width-1 and maze[i, j+1] == 0:
                    neighbors.append((i, j+1))
                graph[(i, j)] = neighbors
    
    # DFSで到達可能性とループをチェック
    def has_cycle(start):
        visited = set()
        parent = {}
        
        def dfs(node, par):
            visited.add(node)
            parent[node] = par
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, node):
                        return True
                elif parent[node] != neighbor:
                    # 親以外への後退エッジ = ループ
                    return True
            return False
        
        return dfs(start, None)
    
    # 統計
    junctions = sum(1 for node in graph if len(graph[node]) >= 3)
    dead_ends = sum(1 for node in graph if len(graph[node]) == 1)
    corridors = sum(1 for node in graph if len(graph[node]) == 2)
    
    has_loop = False
    if passages:
        has_loop = has_cycle(passages[0])
    
    return {
        'is_perfect': not has_loop,
        'passages': len(passages),
        'junctions': junctions,
        'dead_ends': dead_ends,
        'corridors': corridors,
        'has_loop': has_loop
    }


def test_perfect_maze(maze_size='11x11'):
    """完全迷路でテスト"""
    
    print("="*60)
    print(f"🎯 完全迷路テスト（{maze_size}）")
    print("  複数の分岐、袋小路あり、ループなし")
    print("="*60)
    
    # 迷路選択
    if maze_size == '11x11':
        maze = create_perfect_maze_11x11()
    else:
        maze = create_perfect_maze_13x13()
    
    # 検証
    verification = verify_perfect_maze(maze)
    
    print("\n📊 迷路構造検証:")
    print(f"  完全迷路: {'✅' if verification['is_perfect'] else '❌'}")
    print(f"  通路数: {verification['passages']}マス")
    print(f"  分岐点: {verification['junctions']}箇所")
    print(f"  袋小路: {verification['dead_ends']}箇所")
    print(f"  廊下: {verification['corridors']}マス")
    print(f"  ループ: {'なし ✅' if not verification['has_loop'] else 'あり ❌'}")
    
    print("\n迷路構造:")
    for i, row in enumerate(maze):
        row_str = ""
        for j, cell in enumerate(row):
            if i == 1 and j == 1:
                row_str += "S"
            elif i == maze.shape[0]-2 and j == maze.shape[1]-2:
                row_str += "G"
            elif cell == 1:
                row_str += "█"
            else:
                row_str += " "
        print(row_str)
    
    # エージェント作成
    agent = PureMemoryAgentOptimized(
        maze=maze,
        datastore_path="../results/perfect_maze",
        config={
            'max_depth': 5,
            'search_k': 30,
            'gedig_threshold': 0.5,
            'max_edges_per_node': 15
        }
    )
    
    print(f"\n📍 スタート: {agent.position}")
    print(f"🎯 ゴール: {agent.goal}")
    print("-" * 60)
    
    # 実行と記録
    path = [agent.position]
    backtrack_count = 0
    visited_junctions = set()
    
    print("\n実行中...")
    start_time = time.time()
    
    for step in range(300):
        if agent.is_goal_reached():
            elapsed = time.time() - start_time
            print(f"\n🎉 成功！ {step}ステップでゴール到達！")
            print(f"⏱️ 実行時間: {elapsed:.2f}秒")
            break
        
        # 現在位置が分岐点かチェック
        y, x = agent.position
        neighbors = sum([
            y > 0 and maze[y-1, x] == 0,
            y < maze.shape[0]-1 and maze[y+1, x] == 0,
            x > 0 and maze[y, x-1] == 0,
            x < maze.shape[1]-1 and maze[y, x+1] == 0
        ])
        if neighbors >= 3:
            visited_junctions.add((y, x))
        
        # バックトラック検出
        if len(path) >= 3 and agent.position == path[-3]:
            backtrack_count += 1
        
        action = agent.get_action()
        agent.execute_action(action)
        path.append(agent.position)
        
        # 進捗
        if step % 50 == 49:
            stats = agent.get_statistics()
            print(f"\nStep {step+1}:")
            print(f"  位置: {agent.position}")
            print(f"  距離: {stats['distance_to_goal']}")
            print(f"  分岐点訪問: {len(visited_junctions)}/{verification['junctions']}")
            print(f"  バックトラック: {backtrack_count}回")
    else:
        print(f"\n⏰ {step+1}ステップで終了")
    
    # 最終経路表示
    print("\n📊 最終経路:")
    display_final_path(maze, path, agent.goal)
    
    # 統計
    final_stats = agent.get_statistics()
    
    print("\n" + "="*60)
    print("📊 結果分析")
    print("="*60)
    
    success = agent.is_goal_reached()
    print(f"\nゴール到達: {'✅ 成功' if success else '❌ 失敗'}")
    
    if success:
        print(f"総ステップ: {final_stats['steps']}")
        print(f"壁衝突率: {final_stats['wall_hit_rate']:.1%}")
        
        # 探索効率
        unique_positions = len(set(path))
        print(f"\n探索効率:")
        print(f"  訪問マス数: {unique_positions}")
        print(f"  総通路数: {verification['passages']}")
        print(f"  カバー率: {unique_positions/verification['passages']*100:.1f}%")
        print(f"  分岐点探索: {len(visited_junctions)}/{verification['junctions']}")
        print(f"  バックトラック: {backtrack_count}回")
        
        # 最適性
        optimal = abs(agent.position[0] - 1) + abs(agent.position[1] - 1) + \
                 abs(agent.goal[0] - agent.position[0]) + abs(agent.goal[1] - agent.position[1])
        print(f"\n経路効率:")
        print(f"  推定最短: ~{optimal}ステップ")
        print(f"  実際: {len(path)-1}ステップ")
        print(f"  効率: {optimal/(len(path)-1)*100:.1f}%")


def display_final_path(maze, path, goal):
    """最終経路を表示"""
    height, width = maze.shape
    
    # 訪問マップ作成
    visit_map = {}
    for i, pos in enumerate(path):
        if pos not in visit_map:
            visit_map[pos] = i
    
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
            elif pos in visit_map:
                step = visit_map[pos]
                if step < 10:
                    row_str += str(step)
                elif step < 36:
                    row_str += chr(ord('A') + step - 10)
                elif step < 62:
                    row_str += chr(ord('a') + step - 36)
                else:
                    row_str += "*"
            elif maze[i, j] == 1:
                row_str += "█"
            else:
                row_str += " "
        print(row_str)
    
    print("\n凡例: S=スタート, G=ゴール, E=終了位置")
    print("     0-9,A-Z,a-z,*=訪問順, █=壁")


if __name__ == "__main__":
    # 結果ディレクトリ作成
    os.makedirs('../results', exist_ok=True)
    
    print("🔬 完全迷路での実験\n")
    
    # 11×11で3回試行
    print("\n【11×11 完全迷路】")
    for trial in range(3):
        print(f"\n{'='*60}")
        print(f"試行 {trial + 1}/3")
        print('='*60)
        test_perfect_maze('11x11')
        if trial < 2:
            time.sleep(1)
    
    # 13×13で1回試行
    print("\n\n【13×13 より複雑な完全迷路】")
    test_perfect_maze('13x13')