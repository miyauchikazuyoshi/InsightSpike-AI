#!/usr/bin/env python3
"""
ランダムウォークベースライン
"""

import numpy as np
from test_true_perfect_maze import generate_perfect_maze_dfs


def random_walk_baseline(maze_size=(11, 11), max_steps=500, seed=42):
    """ランダムウォークのベースライン"""
    maze = generate_perfect_maze_dfs(maze_size, seed=seed)
    height, width = maze.shape
    position = (1, 1)
    goal = (height - 2, width - 2)
    
    actions = ['up', 'right', 'down', 'left']
    action_deltas = {
        'up': (-1, 0), 'right': (0, 1),
        'down': (1, 0), 'left': (0, -1)
    }
    
    wall_hits = 0
    
    for step in range(max_steps):
        if position == goal:
            return True, step, wall_hits / max(1, step)
        
        # ランダムに行動選択
        action = np.random.choice(actions)
        dx, dy = action_deltas[action]
        new_x = position[0] + dx
        new_y = position[1] + dy
        
        if (0 <= new_x < height and 
            0 <= new_y < width and 
            maze[new_x, new_y] == 0):
            position = (new_x, new_y)
        else:
            wall_hits += 1
    
    return False, max_steps, wall_hits / max_steps


if __name__ == "__main__":
    print("ランダムウォークベースライン（100回実行）")
    print("="*50)
    
    successes = 0
    total_steps = []
    wall_hit_rates = []
    
    for i in range(100):
        success, steps, wall_rate = random_walk_baseline(seed=42+i)
        if success:
            successes += 1
            total_steps.append(steps)
        wall_hit_rates.append(wall_rate)
    
    print(f"成功率: {successes}/100 = {successes}%")
    if total_steps:
        print(f"平均ステップ数（成功時）: {np.mean(total_steps):.0f}")
    print(f"平均壁衝突率: {np.mean(wall_hit_rates)*100:.1f}%")