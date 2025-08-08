#!/usr/bin/env python3
"""
Simple True geDIG Test
======================

Test true geDIG on small mazes to verify it works.
"""

from true_pure_gedig_navigator import TruePureGeDIGNavigator
from pure_episodic_navigator import create_complex_maze, visualize_maze_with_path
import time

# Test on 10x10
size = 10
maze = create_complex_maze(size, seed=42)

print("="*70)
print(f"TRUE geDIG TEST - {size}×{size} maze")
print("="*70)
print("\n本来のgeDIG定義による純粋な実装:")
print("- geDIG = GED - IG を最小化")
print("- 探索ボーナスなし、訪問ペナルティなし")
print("- 純粋に情報理論的なナビゲーション\n")

nav = TruePureGeDIGNavigator(maze)

start_time = time.time()
steps = 0
max_steps = 500

# Track decisions
action_history = []
gedig_values = []

while nav.position != nav.goal and steps < max_steps:
    if steps % 50 == 0:
        dist = abs(nav.position[0] - nav.goal[0]) + abs(nav.position[1] - nav.goal[1])
        print(f"Step {steps}: pos={nav.position}, dist={dist}")
    
    # Get action with geDIG values
    visual = nav.visual_memory.get(nav.position, {})
    action_gedigs = {}
    
    for action in ['up', 'right', 'down', 'left']:
        if visual.get(action) != 'wall':
            gedig = nav.evaluate_action_gedig(nav.position, action)
            action_gedigs[action] = gedig
    
    if not action_gedigs:
        break
    
    # Choose minimum geDIG
    best_action = min(action_gedigs.items(), key=lambda x: x[1])[0]
    best_gedig = action_gedigs[best_action]
    
    action_history.append(best_action)
    gedig_values.append(best_gedig)
    
    # Execute
    old_pos = nav.position
    dx, dy = {'up': (0, -1), 'right': (1, 0), 
             'down': (0, 1), 'left': (-1, 0)}[best_action]
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
    
    nav.add_episode(old_pos, best_action, result, reached_goal)
    steps += 1
    
    if reached_goal:
        break

elapsed = time.time() - start_time

if nav.position == nav.goal:
    print(f"\n✓ 成功！{steps}ステップで到達")
    print(f"時間: {elapsed:.1f}秒")
    print(f"効率: {steps / (2 * (size - 2)):.2f}倍（最適解比）")
    
    # Analyze geDIG values
    negative_count = sum(1 for g in gedig_values if g < 0)
    print(f"\ngeDIG値の分析:")
    print(f"- 負のgeDIG行動: {negative_count}/{len(gedig_values)} "
          f"({negative_count/len(gedig_values)*100:.1f}%)")
    print(f"- 最小geDIG: {min(gedig_values):.3f}")
    print(f"- 最大geDIG: {max(gedig_values):.3f}")
    
    # Show last few steps
    print(f"\n最後の5ステップ:")
    for i in range(max(0, len(action_history)-5), len(action_history)):
        print(f"  Step {i}: {action_history[i]}, geDIG={gedig_values[i]:.3f}")
    
    visualize_maze_with_path(maze, nav.path, 'true_gedig_success.png')
    print("\n画像保存: true_gedig_success.png")
else:
    print(f"\n✗ {steps}ステップで未到達")
    print(f"最終位置: {nav.position}")
    print(f"ゴールまでの距離: {abs(nav.position[0] - nav.goal[0]) + abs(nav.position[1] - nav.goal[1])}")

print(f"\n統計:")
print(f"- エピソード数: {len(nav.episodes)}")
print(f"- グラフ: {nav.episode_graph.number_of_nodes()}ノード, "
      f"{nav.episode_graph.number_of_edges()}エッジ")
print(f"- geDIG計算回数: {nav.gedig_calculations}")