#!/usr/bin/env python3
"""
トリッキーな迷路でのテスト
小さいが罠の多い迷路で実験
"""

import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from pure_memory_agent_optimized import PureMemoryAgentOptimized


def create_tricky_11x11_maze():
    """トリッキーな11×11迷路を作成（複数の偽ルートと罠）"""
    maze = np.array([
        [1,1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,1,0,0,0,1],  # 上部に偽ルート
        [1,0,1,1,1,0,1,0,1,0,1],
        [1,0,0,0,1,0,0,0,1,0,1],  # 左側に袋小路
        [1,1,1,0,1,1,1,1,1,0,1],
        [1,0,0,0,0,0,0,0,0,0,1],  # 中央の大きな空間（罠）
        [1,0,1,1,1,1,1,1,1,0,1],
        [1,0,0,0,0,0,0,0,1,0,1],  # 下部の袋小路
        [1,1,1,1,1,1,1,0,1,0,1],
        [1,0,0,0,0,0,0,0,0,0,1],  # 正解ルート
        [1,1,1,1,1,1,1,1,1,1,1]
    ])
    return maze


def test_tricky_maze():
    """トリッキーな迷路でテスト"""
    
    print("="*60)
    print("🎯 トリッキーな迷路テスト（11×11）")
    print("  偽ルート、袋小路、大空間の罠あり")
    print("="*60)
    
    # 迷路生成
    maze = create_tricky_11x11_maze()
    
    print("\n迷路構造（S=スタート, G=ゴール）:")
    for i, row in enumerate(maze):
        row_str = ""
        for j, cell in enumerate(row):
            if i == 1 and j == 1:
                row_str += "S"
            elif i == 9 and j == 9:
                row_str += "G"
            elif cell == 1:
                row_str += "█"
            else:
                row_str += " "
        print(row_str)
    
    # トリッキーな要素の説明
    print("\n⚠️ トリッキーな要素:")
    print("  1. 上部（行1-2）: ゴールから遠ざかる偽ルート")
    print("  2. 中央（行5）: 大きな空間だが行き止まり")
    print("  3. 左側: 複数の袋小路")
    print("  4. 正解: 最下部を通る細い経路")
    
    # エージェント作成
    agent = PureMemoryAgentOptimized(
        maze=maze,
        datastore_path="../results/tricky_maze",
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
    trap_visits = {
        'upper_fake': 0,  # 上部の偽ルート
        'center_void': 0,  # 中央の大空間
        'left_deadend': 0  # 左の袋小路
    }
    
    print("\n実行中...")
    for step in range(200):
        if agent.is_goal_reached():
            print(f"\n🎉 成功！ {step}ステップでゴール到達！")
            break
        
        # 罠への訪問を記録
        y, x = agent.position
        if y <= 2 and x >= 6:
            trap_visits['upper_fake'] += 1
        elif y == 5:
            trap_visits['center_void'] += 1
        elif x <= 3 and y in [3, 4, 7, 8]:
            trap_visits['left_deadend'] += 1
        
        action = agent.get_action()
        agent.execute_action(action)
        path.append(agent.position)
        
        # 進捗
        if step % 40 == 39:
            stats = agent.get_statistics()
            print(f"\nStep {step+1}:")
            print(f"  位置: {agent.position}")
            print(f"  距離: {stats['distance_to_goal']}")
            print(f"  罠訪問: 上部{trap_visits['upper_fake']}回, "
                  f"中央{trap_visits['center_void']}回, "
                  f"左側{trap_visits['left_deadend']}回")
    else:
        print(f"\n⏰ {step+1}ステップで終了")
    
    # 最終経路表示
    print("\n📊 最終経路（数字は訪問順）:")
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
        
        # 罠の分析
        total_trap_visits = sum(trap_visits.values())
        print(f"\n罠への訪問分析:")
        print(f"  上部偽ルート: {trap_visits['upper_fake']}回")
        print(f"  中央大空間: {trap_visits['center_void']}回")
        print(f"  左側袋小路: {trap_visits['left_deadend']}回")
        print(f"  合計: {total_trap_visits}回")
        
        if total_trap_visits < 50:
            print("  → ✨ 罠を効率的に回避！")
        elif total_trap_visits < 100:
            print("  → 🔍 適度な探索で罠を学習")
        else:
            print("  → ⚠️ 罠に多く引っかかった")
        
        # 学習品質
        print(f"\n学習品質:")
        print(f"  平均geDIG: {final_stats['avg_gedig']:.3f}")
        if final_stats['avg_gedig'] < 0:
            print("  → 良好な学習（情報利得 > 編集距離）")
        
        # 深度使用
        print(f"\n深度使用パターン:")
        total = sum(final_stats['depth_usage'].values())
        if total > 0:
            deep = sum(final_stats['depth_usage'].get(d, 0) for d in range(4, 6))
            print(f"  深い推論（4-5ホップ）: {deep/total*100:.1f}%")
            if deep/total > 0.5:
                print("  → 深い推論が罠回避に貢献")


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
                # 訪問順を圧縮表示
                if step < 10:
                    row_str += str(step)
                elif step < 100:
                    row_str += "+"
                else:
                    row_str += "*"
            elif maze[i, j] == 1:
                row_str += "█"
            else:
                row_str += " "
        
        # 行の説明
        if i == 1:
            row_str += "  ← 偽ルート"
        elif i == 5:
            row_str += "  ← 大空間の罠"
        elif i == 9:
            row_str += "  ← 正解ルート"
        
        print(row_str)
    
    print("\n凡例: S=スタート, G=ゴール, 0-9=初期探索")
    print("     +=中期探索, *=後期探索, █=壁")


if __name__ == "__main__":
    # 結果ディレクトリ作成
    os.makedirs('../results', exist_ok=True)
    
    # 3回試行
    print("🔬 3回試行して罠への対処を分析\n")
    
    for trial in range(3):
        print(f"\n{'='*60}")
        print(f"試行 {trial + 1}/3")
        print('='*60)
        
        test_tricky_maze()
        
        if trial < 2:
            print("\n次の試行まで待機...")
            time.sleep(1)