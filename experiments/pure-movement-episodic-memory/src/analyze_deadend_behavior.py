#!/usr/bin/env python3
"""
袋小路でのメッセージパッシング動作を分析
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from pure_memory_agent_optimized import PureMemoryAgentOptimized


def create_simple_deadend_maze():
    """単純な袋小路迷路を作成"""
    maze = np.array([
        [1,1,1,1,1,1,1],
        [1,0,0,0,0,0,1],  # 長い廊下
        [1,1,1,1,1,0,1],  # 袋小路への入口
        [1,0,0,0,0,0,1],  # 袋小路
        [1,1,1,1,1,1,1]
    ])
    return maze


def analyze_deadend_behavior():
    """袋小路での挙動を詳細分析"""
    
    print("="*60)
    print("🔍 袋小路でのメッセージパッシング分析")
    print("="*60)
    
    # 袋小路迷路
    maze = create_simple_deadend_maze()
    
    print("\n迷路構造:")
    for i, row in enumerate(maze):
        row_str = ""
        for j, cell in enumerate(row):
            if i == 1 and j == 1:
                row_str += "S"
            elif i == 1 and j == 5:
                row_str += "G"
            elif cell == 1:
                row_str += "█"
            else:
                row_str += " "
        print(row_str)
    
    print("\n解説:")
    print("  S→→→→→G が正解ルート")
    print("  下の袋小路は罠")
    
    # エージェント作成
    agent = PureMemoryAgentOptimized(
        maze=maze,
        datastore_path="../results/deadend_analysis",
        config={
            'max_depth': 3,
            'search_k': 10,
            'gedig_threshold': 0.5
        }
    )
    
    # 手動で袋小路に入れる
    print("\n実験: 袋小路に入って戻る過程を観察")
    print("-" * 60)
    
    # 袋小路まで移動
    moves = [
        ('right', (1, 2)),
        ('right', (1, 3)),
        ('right', (1, 4)),
        ('down', (2, 5)),  # 袋小路への入口
        ('down', (3, 5)),  # 袋小路の奥
    ]
    
    for i, (action, expected_pos) in enumerate(moves):
        print(f"\nStep {i+1}: {action}")
        
        # 行動実行
        success = agent.execute_action(action)
        
        if success:
            print(f"  位置: {agent.position} → {expected_pos}")
            
            # 現在のエピソード情報
            if agent.experience_metadata:
                latest = agent.experience_metadata[-1]
                print(f"  最新エピソード: {latest.get('type', 'unknown')}")
        else:
            print(f"  壁に衝突！")
    
    # 袋小路での推論を観察
    print("\n" + "="*60)
    print("🧠 袋小路での推論プロセス")
    print("="*60)
    
    # クエリベクトル
    query = agent._create_task_query()
    print(f"\nクエリベクトル: {query}")
    
    # 類似エピソード検索
    distances, indices = agent.vector_index.search(
        query.reshape(1, -1),
        k=10
    )
    
    print("\n類似エピソード Top 5:")
    for rank, (dist, idx) in enumerate(zip(distances[0][:5], indices[0][:5])):
        if 0 <= idx < len(agent.experience_metadata):
            meta = agent.experience_metadata[idx]
            print(f"  {rank+1}. 位置{meta.get('pos', '?')}, "
                  f"方向={meta.get('direction', '?')}, "
                  f"壁={meta.get('is_wall', '?')}, "
                  f"類似度={dist:.3f}")
    
    # メッセージパッシング前後の比較
    print("\n各深度でのメッセージパッシング結果:")
    
    for depth in range(1, 4):
        agent.stats['depth_usage'][depth] = agent.stats['depth_usage'].get(depth, 0) + 1
        
        # メッセージパッシング実行
        aggregated = agent._message_passing(indices[0][:10].tolist(), depth)
        
        print(f"\n深度{depth}:")
        print(f"  位置成分: ({aggregated[0]:.2f}, {aggregated[1]:.2f})")
        print(f"  方向成分: {aggregated[2]:.2f}")
        print(f"  成功/失敗: {aggregated[3]:.2f}")
        print(f"  壁情報: {aggregated[4]:.2f}")
        
        # 方向の解釈
        direction_idx = int(aggregated[2] * 3)
        directions = ['up', 'right', 'down', 'left']
        if 0 <= direction_idx < 4:
            print(f"  → 推奨方向: {directions[direction_idx]}")
    
    # 実際の行動選択
    print("\n" + "="*60)
    print("💡 実際の行動選択")
    print("="*60)
    
    for step in range(10):
        print(f"\nStep {step+1}:")
        print(f"  現在位置: {agent.position}")
        
        action = agent.get_action()
        print(f"  選択した行動: {action}")
        
        success = agent.execute_action(action)
        print(f"  結果: {'成功' if success else '壁衝突'}")
        
        # 袋小路から脱出できたか
        if agent.position[0] < 3:  # 袋小路から出た
            print("  ✅ 袋小路から脱出！")
            break
    else:
        print("  ❌ 袋小路から脱出できず")
    
    # 統計
    print("\n" + "="*60)
    print("📊 分析結果")
    print("="*60)
    
    stats = agent.get_statistics()
    print(f"\n基本統計:")
    print(f"  総ステップ: {stats['steps']}")
    print(f"  壁衝突率: {stats['wall_hit_rate']:.1%}")
    print(f"  平均geDIG: {stats['avg_gedig']:.3f}")
    
    # メッセージパッシングの問題点
    print("\n💭 観察された問題点:")
    print("  1. 方向情報が平均化されて曖昧になる")
    print("  2. 「戻る」という明確な信号が生成されにくい")
    print("  3. 複数の失敗経験が混ざり合う")
    
    return agent


if __name__ == "__main__":
    # 結果ディレクトリ作成
    os.makedirs('../results', exist_ok=True)
    
    # 分析実行
    agent = analyze_deadend_behavior()
    
    print("\n" + "="*60)
    print("🔬 提案: 方向反転を考慮した改良")
    print("="*60)
    print("""
    1. 袋小路検出時に「反対方向」を明示的に生成
    2. 失敗経験に「逆方向推奨」フラグを追加
    3. メッセージパッシングで方向を反転考慮
    
    例：
    - 南に進んで失敗 → 北を推奨
    - 東に進んで袋小路 → 西を推奨
    """)