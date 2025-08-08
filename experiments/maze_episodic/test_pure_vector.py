#!/usr/bin/env python3
"""
Pure Vector-based geDIG Test (No Cheating)
==========================================

Test navigation using ONLY episode similarity and message passing.
No exploration bonus, no revisit penalty.
"""

from donut_gedig_navigator_simple import DonutGeDIGNavigator
from pure_episodic_navigator import create_complex_maze, visualize_maze_with_path
import time

print("="*70)
print("純粋なベクトル検索によるナビゲーション（チートなし）")
print("="*70)
print("\n探索ボーナスや再訪問ペナルティを使わず、")
print("エピソード類似度とメッセージパッシングのみで迷路を解く\n")

# Test on 10x10 first
size = 10
maze = create_complex_maze(size, seed=42)

nav = DonutGeDIGNavigator(maze, inner_radius=0.1, outer_radius=0.7)

print(f"Testing {size}×{size} maze...")
print(f"Start: {nav.position}, Goal: {nav.goal}")

start = time.time()
steps = 0
max_steps = 1000

while nav.position != nav.goal and steps < max_steps:
    if steps % 50 == 0:
        dist = abs(nav.position[0] - nav.goal[0]) + abs(nav.position[1] - nav.goal[1])
        print(f"Step {steps}: pos={nav.position}, dist={dist}, episodes={len(nav.episodes)}")
    
    action = nav.decide_action()
    
    old_pos = nav.position
    dx, dy = {'up': (0, -1), 'right': (1, 0), 
             'down': (0, 1), 'left': (-1, 0)}[action]
    new_pos = (nav.position[0] + dx, nav.position[1] + dy)
    
    result = 'wall'
    reached_goal = False
    
    if (0 <= new_pos[0] < nav.width and 
        0 <= new_pos[1] < nav.height and
        nav.maze[new_pos[1], new_pos[0]] == 0):
        
        if new_pos in nav.visited:
            result = 'visited'
        else:
            result = 'success'
        
        nav.position = new_pos
        nav.visited.add(new_pos)
        nav.path.append(new_pos)
        nav._update_visual_memory(new_pos[0], new_pos[1])
        
        if new_pos == nav.goal:
            reached_goal = True
    
    nav.add_episode(old_pos, action, result, reached_goal)
    steps += 1
    
    if reached_goal:
        break

elapsed = time.time() - start

if nav.position == nav.goal:
    print(f"\n✓ 成功！ {steps}ステップで到達")
    print(f"時間: {elapsed:.1f}秒")
    print(f"効率: {steps / (2 * (size - 2)):.2f}倍（最適解比）")
    
    # Analyze pure geDIG behavior
    print(f"\n純粋なgeDIG統計:")
    print(f"- エピソード数: {len(nav.episodes)}")
    print(f"- グラフエッジ数: {sum(len(n) for n in nav.graph.values()) // 2}")
    
    # Result distribution
    results = {'success': 0, 'wall': 0, 'visited': 0}
    for ep in nav.episodes:
        results[ep['result']] += 1
    
    print(f"\n行動結果の分布:")
    for result, count in results.items():
        print(f"- {result}: {count} ({count/len(nav.episodes)*100:.1f}%)")
    
    print(f"\nポイント:")
    print(f"1. 探索ボーナスなしでも迷路を解けた")
    print(f"2. 再訪問は自然に発生（{results['visited']}回）")
    print(f"3. ゴール到達情報がグラフを通じて伝播")
    
    visualize_maze_with_path(maze, nav.path, 'pure_vector_10x10.png')
else:
    print(f"\n✗ {steps}ステップで未到達")
    print(f"最終位置: {nav.position}")
    print(f"ゴールまでの距離: {abs(nav.position[0] - nav.goal[0]) + abs(nav.position[1] - nav.goal[1])}")