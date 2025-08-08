#!/usr/bin/env python3
"""Check true geDIG behavior"""

from true_pure_gedig_navigator import TruePureGeDIGNavigator
from pure_episodic_navigator import create_complex_maze

# 5x5 maze
maze = create_complex_maze(5, seed=42)
nav = TruePureGeDIGNavigator(maze)

print("True geDIG挙動チェック (5×5)")
print("="*40)

# Track positions
position_history = []

for step in range(20):
    print(f"\nStep {step}: pos={nav.position}")
    position_history.append(nav.position)
    
    # Check for cycles
    if len(position_history) >= 4:
        recent = position_history[-4:]
        if recent[0] == recent[2] and recent[1] == recent[3]:
            print(f"循環検出: {recent[0]} ↔ {recent[1]}")
            break
    
    visual = nav.visual_memory.get(nav.position, {})
    action_gedigs = {}
    
    for action in ['up', 'right', 'down', 'left']:
        if visual.get(action) != 'wall':
            gedig = nav.evaluate_action_gedig(nav.position, action)
            action_gedigs[action] = gedig
    
    print(f"geDIG値: {action_gedigs}")
    
    if not action_gedigs:
        break
    
    # Choose minimum
    best_action = min(action_gedigs.items(), key=lambda x: x[1])[0]
    print(f"選択: {best_action}")
    
    # Execute
    dx, dy = {'up': (0, -1), 'right': (1, 0), 
             'down': (0, 1), 'left': (-1, 0)}[best_action]
    new_pos = (nav.position[0] + dx, nav.position[1] + dy)
    
    if (0 <= new_pos[0] < 5 and 0 <= new_pos[1] < 5 and 
        maze[new_pos[1], new_pos[0]] == 0):
        nav.position = new_pos
        nav.visited.add(new_pos)
        nav._update_visual_memory(new_pos[0], new_pos[1])
        result = 'visited' if new_pos in nav.visited else 'success'
    else:
        result = 'wall'
    
    nav.add_episode(nav.position, best_action, result, nav.position == nav.goal)
    
    if nav.position == nav.goal:
        print("\nゴール到達!")
        break

print(f"\n分析:")
print(f"1. 純粋なgeDIG最小化は局所最適に陥りやすい")
print(f"2. 全ての方向が似たようなgeDIG値になる")
print(f"3. ゴール情報がない初期段階では探索が不十分")