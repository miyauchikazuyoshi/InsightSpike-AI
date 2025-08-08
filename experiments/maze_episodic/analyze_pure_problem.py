#!/usr/bin/env python3
"""
純粋なgeDIG評価の問題分析
========================
"""

from donut_gedig_navigator_simple import DonutGeDIGNavigator
from pure_episodic_navigator import create_complex_maze

# 小さな迷路でテスト
size = 5
maze = create_complex_maze(size, seed=42)

nav = DonutGeDIGNavigator(maze, inner_radius=0.1, outer_radius=0.8)

print(f"{size}×{size}迷路で問題を分析")
print(f"Start: {nav.position}, Goal: {nav.goal}")

# 20ステップ実行して挙動を観察
visited_sequence = []

for step in range(20):
    print(f"\nStep {step}: pos={nav.position}")
    
    # 各アクションのスコアを詳細に見る
    visual = nav.visual_memory.get(nav.position, {})
    action_scores = {}
    
    for action in ['up', 'right', 'down', 'left']:
        if visual.get(action) == 'wall':
            action_scores[action] = -10.0
        else:
            score, hop = nav.evaluate_action(nav.position, action)
            action_scores[action] = score
            
    print(f"アクションスコア: {action_scores}")
    
    # 最高スコアのアクションを選択
    best_action = max(action_scores.items(), key=lambda x: x[1])[0]
    print(f"選択: {best_action}")
    
    # 実行
    old_pos = nav.position
    dx, dy = {'up': (0, -1), 'right': (1, 0), 
             'down': (0, 1), 'left': (-1, 0)}[best_action]
    new_pos = (nav.position[0] + dx, nav.position[1] + dy)
    
    result = 'wall'
    if (0 <= new_pos[0] < nav.width and 
        0 <= new_pos[1] < nav.height and
        nav.maze[new_pos[1], new_pos[0]] == 0):
        nav.position = new_pos
        result = 'visited' if new_pos in nav.visited else 'success'
        nav.visited.add(new_pos)
        nav._update_visual_memory(new_pos[0], new_pos[1])
    
    nav.add_episode(old_pos, best_action, result, nav.position == nav.goal)
    visited_sequence.append(nav.position)
    
    # 循環検出
    if len(visited_sequence) >= 4:
        recent = visited_sequence[-4:]
        if recent[0] == recent[2] and recent[1] == recent[3]:
            print(f"\n循環検出！ {recent[0]} ↔ {recent[1]}")
            break

print(f"\n分析結果:")
print(f"1. エピソード数: {len(nav.episodes)}")
print(f"2. 訪問位置数: {len(nav.visited)}")
print(f"3. 問題: 純粋な類似度評価では局所的な循環に陥る")
print(f"4. 原因: すべての未知の方向が同じスコアになる")
print(f"5. 解決策が必要: ランダム性の導入、または別の仕組み")