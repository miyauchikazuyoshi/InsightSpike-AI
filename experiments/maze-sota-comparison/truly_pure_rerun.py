#!/usr/bin/env python3
"""
Truly Pure Navigator - Same Maze Twice
======================================

Test learning by solving the same maze twice.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import time
import matplotlib.pyplot as plt

@dataclass 
class Experience:
    """Simple experience record"""
    state: Tuple[float, float, float]  # (x, y, action)
    reward: float  # +1 for move, -1 for wall
    
class TrulyPureNavigator:
    """Navigator that learns purely from experience"""
    
    def __init__(self, maze: np.ndarray):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        self.experiences: List[Experience] = []
        
        # Statistics
        self.wall_hits = 0
        self.moves = 0
        self.step_count = 0
        
        # For tracking learning
        self.wall_hit_history = []
        self.position_history = []
        
    def reset_position(self):
        """Reset position but keep experiences"""
        self.position = (1, 1)
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        self.wall_hits = 0
        self.moves = 0
        # Keep experiences for learning!
        
    def get_state(self, pos: Tuple[int, int], action: int) -> Tuple[float, float, float]:
        """Convert position and action to state"""
        return (pos[0] / self.width, pos[1] / self.height, action / 3.0)
    
    def get_action_value(self, pos: Tuple[int, int], action: int) -> float:
        """Get value for state-action pair based on experience"""
        state = self.get_state(pos, action)
        
        # No experience yet
        if not self.experiences:
            return 0.0
        
        # Find similar experiences
        total_weight = 0.0
        weighted_reward = 0.0
        
        for exp in self.experiences:
            # Simple distance metric
            dist = sum((s1 - s2)**2 for s1, s2 in zip(state, exp.state))**0.5
            weight = np.exp(-dist * 10)  # Similarity weight
            
            total_weight += weight
            weighted_reward += weight * exp.reward
        
        if total_weight > 0:
            return weighted_reward / total_weight
        return 0.0
    
    def navigate(self, goal: Tuple[int, int], max_steps: int = 5000, run_number: int = 1) -> Dict:
        """Navigate learning from experience"""
        print(f"\n{'='*50}")
        print(f"Run {run_number}: {'WITH PRIOR EXPERIENCE' if run_number > 1 else 'FIRST TIME'}")
        print(f"Start: {self.position}, Goal: {goal}")
        print(f"Prior experiences: {len(self.experiences)}")
        
        steps = 0
        start_time = time.time()
        
        # Track statistics
        wall_hits_per_100 = []
        
        while self.position != goal and steps < max_steps:
            # Track wall hits per 100 steps
            if steps % 100 == 0:
                if steps > 0:
                    recent_hits = len([h for h in self.wall_hit_history[-100:] if h])
                    wall_hits_per_100.append(recent_hits)
                    
            # Progress
            if steps % 500 == 0 and steps > 0:
                dist = abs(self.position[0] - goal[0]) + abs(self.position[1] - goal[1])
                hit_rate = self.wall_hits / (self.moves + self.wall_hits) * 100 if (self.moves + self.wall_hits) > 0 else 0
                print(f"Step {steps}: pos={self.position}, dist={dist}, "
                      f"wall_hit_rate={hit_rate:.1f}%, total_exp={len(self.experiences)}")
            
            # Evaluate all actions
            action_values = []
            for action in range(4):  # 0=up, 1=right, 2=down, 3=left
                value = self.get_action_value(self.position, action)
                action_values.append(value)
            
            # Softmax selection
            values = np.array(action_values)
            
            # Less exploration noise for second run
            noise_scale = 0.1 if run_number == 1 else 0.05
            values += np.random.normal(0, noise_scale, 4)
            
            probs = np.exp(values)
            probs = probs / probs.sum()
            action = np.random.choice(4, p=probs)
            
            # Execute action
            old_pos = self.position
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
            new_pos = (self.position[0] + dx, self.position[1] + dy)
            
            # Try move (check bounds and walls for outcome only)
            hit_wall = False
            if (0 <= new_pos[0] < self.width and 
                0 <= new_pos[1] < self.height and
                self.maze[new_pos[1], new_pos[0]] == 0):
                # Success
                reward = 1.0
                self.position = new_pos
                self.visited.add(new_pos)
                self.path.append(new_pos)
                self.moves += 1
            else:
                # Hit wall
                reward = -1.0
                self.wall_hits += 1
                hit_wall = True
            
            # Record experience (from old position)
            exp = Experience(
                state=self.get_state(old_pos, action),
                reward=reward
            )
            self.experiences.append(exp)
            self.wall_hit_history.append(hit_wall)
            self.position_history.append(self.position)
            
            steps += 1
            self.step_count += 1
        
        success = self.position == goal
        elapsed = time.time() - start_time
        
        print(f"\nComplete: success={success}, steps={steps}, wall_hits={self.wall_hits}")
        print(f"Final wall hit rate: {self.wall_hits/(self.moves+self.wall_hits)*100:.1f}%")
        print(f"Path efficiency: {len(set(self.path))/steps*100:.1f}%")
        
        return {
            'success': success,
            'steps': steps,
            'wall_hits': self.wall_hits,
            'moves': self.moves,
            'wall_hit_rate': self.wall_hits / (self.moves + self.wall_hits) * 100,
            'path_efficiency': len(set(self.path)) / steps * 100,
            'time': elapsed,
            'wall_hits_per_100': wall_hits_per_100,
            'unique_positions': len(set(self.path))
        }


def test_same_maze_twice():
    """Test solving the same maze twice"""
    
    # Create a smaller 15x15 maze for faster testing
    np.random.seed(42)
    size = 15
    maze = np.ones((size, size), dtype=int)
    
    # Maze generation
    def carve(x, y):
        maze[y, x] = 0
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        np.random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < size-1 and 0 < ny < size-1 and maze[ny, nx] == 1:
                maze[y + dy//2, x + dx//2] = 0
                maze[ny, nx] = 0
                carve(nx, ny)
    
    carve(1, 1)
    maze[size-2, size-2] = 0
    
    # Add some loops
    for _ in range(10):
        x = np.random.randint(2, size-2)
        y = np.random.randint(2, size-2)
        if maze[y, x] == 1:
            neighbors = sum(1 for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]
                          if 0 <= x+dx < size and 0 <= y+dy < size and maze[y+dy, x+dx] == 0)
            if neighbors >= 2:
                maze[y, x] = 0
    
    print("="*60)
    print("TRULY PURE LEARNING - SAME MAZE TWICE")
    print("="*60)
    
    # Create navigator
    nav = TrulyPureNavigator(maze)
    goal = (size-2, size-2)
    
    # First run
    result1 = nav.navigate(goal, max_steps=5000, run_number=1)
    
    # Reset position but keep experiences
    nav.reset_position()
    
    # Second run
    result2 = nav.navigate(goal, max_steps=5000, run_number=2)
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    print(f"\n{'Metric':<20} {'First Run':>15} {'Second Run':>15} {'Improvement':>15}")
    print("-" * 65)
    
    metrics = [
        ('Steps', 'steps', True),
        ('Wall Hits', 'wall_hits', True),
        ('Wall Hit Rate %', 'wall_hit_rate', True),
        ('Path Efficiency %', 'path_efficiency', False),
        ('Unique Positions', 'unique_positions', True),
        ('Time (seconds)', 'time', True)
    ]
    
    for name, key, lower_is_better in metrics:
        val1 = result1[key]
        val2 = result2[key]
        
        if lower_is_better:
            improvement = (val1 - val2) / val1 * 100 if val1 > 0 else 0
        else:
            improvement = (val2 - val1) / val1 * 100 if val1 > 0 else 0
            
        print(f"{name:<20} {val1:>15.1f} {val2:>15.1f} {improvement:>14.1f}%")
    
    # Plot wall hit rate over time
    if result1['wall_hits_per_100'] and result2['wall_hits_per_100']:
        plt.figure(figsize=(10, 6))
        
        x1 = list(range(len(result1['wall_hits_per_100'])))
        x2 = list(range(len(result2['wall_hits_per_100'])))
        
        plt.plot(x1, result1['wall_hits_per_100'], 'r-', label='First Run', linewidth=2)
        plt.plot(x2, result2['wall_hits_per_100'], 'g-', label='Second Run', linewidth=2)
        
        plt.xlabel('Time (hundreds of steps)')
        plt.ylabel('Wall Hits per 100 Steps')
        plt.title('Learning Effect: Wall Hit Rate Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('truly_pure_learning_curve.png', dpi=150)
        plt.close()
        
        print(f"\nLearning curve saved to truly_pure_learning_curve.png")
    
    return result1, result2


if __name__ == "__main__":
    result1, result2 = test_same_maze_twice()