#!/usr/bin/env python3
"""
中程度に複雑な迷路でのテスト（13×13）
"""

import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from pure_memory_agent_optimized import PureMemoryAgentOptimized


def create_moderate_complex_maze():
    """中程度に複雑な13×13迷路を手動で作成"""
    maze = np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,1,0,0,0,0,0,0,0,1],
        [1,0,1,0,1,0,1,1,1,1,1,0,1],
        [1,0,1,0,0,0,0,0,0,0,1,0,1],
        [1,0,1,1,1,1,1,1,1,0,1,0,1],
        [1,0,0,0,0,0,0,0,1,0,0,0,1],
        [1,1,1,0,1,1,1,0,1,1,1,0,1],
        [1,0,0,0,0,0,1,0,0,0,0,0,1],
        [1,0,1,1,1,0,1,1,1,1,1,0,1],
        [1,0,0,0,1,0,0,0,0,0,1,0,1],
        [1,1,1,0,1,1,1,1,1,0,1,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1]
    ])
    return maze


def test_moderate_maze():
    """中程度の複雑さの迷路でテスト"""
    
    print("="*70)
    print("🌀 中程度に複雑な迷路テスト（13×13）")
    print("  複数の経路と袋小路を含む")
    print("="*70)
    
    # 迷路生成
    maze = create_moderate_complex_maze()
    
    print("\n迷路構造:")
    for i, row in enumerate(maze):
        row_str = ""
        for j, cell in enumerate(row):
            if i == 1 and j == 1:
                row_str += "S"  # スタート
            elif i == 11 and j == 11:
                row_str += "G"  # ゴール
            elif cell == 1:
                row_str += "█"
            else:
                row_str += " "
        print(row_str)
    
    # 迷路の分析
    analyze_maze(maze)
    
    # エージェント作成
    agent = PureMemoryAgentOptimized(
        maze=maze,
        datastore_path="../results/moderate_complex",
        config={
            'max_depth': 6,
            'search_k': 40,
            'gedig_threshold': 0.5,
            'max_edges_per_node': 20
        }
    )
    
    initial_distance = abs(agent.position[0] - agent.goal[0]) + \
                      abs(agent.position[1] - agent.goal[1])
    
    print(f"\n📍 スタート: {agent.position}")
    print(f"🎯 ゴール: {agent.goal}")
    print(f"📏 初期距離: {initial_distance}")
    print("-" * 70)
    
    # 実行
    path = [agent.position]
    visited_positions = set()
    wall_hit_positions = []
    
    print("\n実行中...")
    start_time = time.time()
    
    for step in range(300):
        if agent.is_goal_reached():
            elapsed = time.time() - start_time
            print(f"\n🎉 成功！ {step}ステップでゴール到達！")
            print(f"⏱️ 実行時間: {elapsed:.2f}秒")
            break
        
        prev_pos = agent.position
        action = agent.get_action()
        success = agent.execute_action(action)
        
        if not success:
            # 壁衝突位置を記録
            dx, dy = agent.action_deltas[action]
            wall_pos = (prev_pos[0] + dx, prev_pos[1] + dy)
            wall_hit_positions.append(wall_pos)
        
        path.append(agent.position)
        visited_positions.add(agent.position)
        
        # 進捗報告
        if step % 50 == 49:
            stats = agent.get_statistics()
            print(f"\nStep {step+1}:")
            print(f"  位置: {agent.position}")
            print(f"  距離: {stats['distance_to_goal']}")
            print(f"  訪問済み: {len(visited_positions)}マス")
            print(f"  壁衝突率: {stats['wall_hit_rate']:.1%}")
    else:
        print(f"\n⏰ {step+1}ステップで終了")
    
    # 経路をASCIIで表示
    print("\n📊 最終経路:")
    display_path_on_maze(maze, path, agent.goal, visited_positions)
    
    # 統計
    final_stats = agent.get_statistics()
    
    print("\n" + "="*70)
    print("📊 結果分析")
    print("="*70)
    
    print(f"\n基本統計:")
    print(f"  ゴール到達: {'✅ 成功' if agent.is_goal_reached() else f'❌ 距離 {final_stats["distance_to_goal"]}'}")
    print(f"  総ステップ: {final_stats['steps']}")
    print(f"  壁衝突率: {final_stats['wall_hit_rate']:.1%}")
    
    print(f"\n探索効率:")
    print(f"  訪問マス数: {len(visited_positions)}")
    print(f"  総通路数: {np.sum(maze == 0)}")
    print(f"  探索カバー率: {len(visited_positions) / np.sum(maze == 0) * 100:.1f}%")
    
    print(f"\n学習品質:")
    print(f"  平均geDIG: {final_stats['avg_gedig']:.3f}")
    print(f"  グラフエッジ数: {final_stats['graph_edges']}")
    
    # 深度使用分析
    print(f"\n深度使用:")
    total_depth = sum(final_stats['depth_usage'].values())
    if total_depth > 0:
        for depth in sorted(final_stats['depth_usage'].keys()):
            count = final_stats['depth_usage'][depth]
            if count > 0:
                ratio = count / total_depth * 100
                bar = '█' * int(ratio / 5)
                print(f"  {depth}ホップ: {bar} {ratio:.1f}%")
    
    # 経路効率
    if agent.is_goal_reached():
        optimal = initial_distance
        actual = len(path) - 1
        print(f"\n経路効率:")
        print(f"  最適: {optimal}マス")
        print(f"  実際: {actual}ステップ")
        print(f"  効率: {optimal/actual*100:.1f}%")


def analyze_maze(maze):
    """迷路の複雑さを分析"""
    height, width = maze.shape
    
    # 分岐点と袋小路を数える
    junctions = 0
    dead_ends = 0
    corridors = 0
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            if maze[i, j] == 0:
                neighbors = 0
                if maze[i-1, j] == 0: neighbors += 1
                if maze[i+1, j] == 0: neighbors += 1
                if maze[i, j-1] == 0: neighbors += 1
                if maze[i, j+1] == 0: neighbors += 1
                
                if neighbors >= 3:
                    junctions += 1
                elif neighbors == 1:
                    dead_ends += 1
                elif neighbors == 2:
                    corridors += 1
    
    total_passages = np.sum(maze == 0)
    
    print(f"\n📊 迷路分析:")
    print(f"  通路: {total_passages}マス")
    print(f"  分岐点: {junctions}箇所")
    print(f"  袋小路: {dead_ends}箇所")
    print(f"  直線通路: {corridors}箇所")
    print(f"  複雑度: {(junctions*2 + dead_ends) / total_passages * 100:.1f}")


def display_path_on_maze(maze, path, goal, visited):
    """経路を迷路上に表示"""
    height, width = maze.shape
    
    # 経路辞書を作成
    path_dict = {}
    for i, pos in enumerate(path):
        if pos not in path_dict:
            path_dict[pos] = i
    
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
            elif pos in path_dict:
                # 訪問順を表示
                step = path_dict[pos]
                if step < 10:
                    row_str += str(step)
                elif step < 36:
                    row_str += chr(ord('A') + step - 10)
                else:
                    row_str += "*"
            elif pos in visited:
                row_str += "·"
            elif maze[i, j] == 1:
                row_str += "█"
            else:
                row_str += " "
        print(row_str)
    
    print("\n凡例: S=スタート, G=ゴール, E=終了位置")
    print("     0-9,A-Z,*=訪問順, ·=訪問済み, █=壁")


if __name__ == "__main__":
    # 結果ディレクトリ作成
    os.makedirs('../results', exist_ok=True)
    
    # テスト実行
    test_moderate_maze()