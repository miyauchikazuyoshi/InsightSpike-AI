#!/usr/bin/env python3
"""
Truly Pure Simple Navigator
===========================

Simplified version - no if statements for walls, learns from experience.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import time

@dataclass 
class Experience:
    """Simple experience record"""
    state: Tuple[float, float, float]  # (x, y, action)
    reward: float  # +1 for move, -1 for wall
    
class TrulyPureSimple:
    """Simplest possible pure learning navigator"""
    
    def __init__(self, maze: np.ndarray):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.visited = {(1, 1)}
        self.experiences: List[Experience] = []
        
        # Statistics
        self.wall_hits = 0
        self.moves = 0
        
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
    
    def navigate(self, goal: Tuple[int, int], max_steps: int = 5000) -> Dict:
        """Navigate learning from experience"""
        print(f"\nTruly Pure Simple Navigation")
        print(f"Start: {self.position}, Goal: {goal}")
        
        steps = 0
        start_time = time.time()
        
        while self.position != goal and steps < max_steps:
            # Progress
            if steps % 500 == 0 and steps > 0:
                dist = abs(self.position[0] - goal[0]) + abs(self.position[1] - goal[1])
                hit_rate = self.wall_hits / (self.moves + self.wall_hits) * 100 if (self.moves + self.wall_hits) > 0 else 0
                print(f"Step {steps}: pos={self.position}, dist={dist}, "
                      f"wall_hit_rate={hit_rate:.1f}%")
            
            # Evaluate all actions
            action_values = []
            for action in range(4):  # 0=up, 1=right, 2=down, 3=left
                value = self.get_action_value(self.position, action)
                action_values.append(value)
            
            # Softmax selection
            values = np.array(action_values)
            # Add exploration noise
            values += np.random.normal(0, 0.1, 4)
            
            probs = np.exp(values)
            probs = probs / probs.sum()
            action = np.random.choice(4, p=probs)
            
            # Execute action
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
            new_pos = (self.position[0] + dx, self.position[1] + dy)
            
            # Try move (check bounds and walls for outcome only)
            if (0 <= new_pos[0] < self.width and 
                0 <= new_pos[1] < self.height and
                self.maze[new_pos[1], new_pos[0]] == 0):
                # Success
                reward = 1.0
                self.position = new_pos
                self.visited.add(new_pos)
                self.moves += 1
            else:
                # Hit wall
                reward = -1.0
                self.wall_hits += 1
            
            # Record experience
            exp = Experience(
                state=self.get_state(self.position, action),
                reward=reward
            )
            self.experiences.append(exp)
            
            steps += 1
        
        success = self.position == goal
        elapsed = time.time() - start_time
        
        print(f"\nComplete: success={success}, steps={steps}, wall_hits={self.wall_hits}")
        print(f"Final wall hit rate: {self.wall_hits/(self.moves+self.wall_hits)*100:.1f}%")
        
        return {
            'success': success,
            'steps': steps,
            'wall_hits': self.wall_hits,
            'moves': self.moves,
            'wall_hit_rate': self.wall_hits / (self.moves + self.wall_hits) * 100,
            'time': elapsed
        }


# Test on small maze
def test_simple():
    # Small 15x15 maze for faster testing
    np.random.seed(42)
    size = 15
    maze = np.ones((size, size), dtype=int)
    
    # Simple maze generation
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
    
    # Run multiple times to see learning
    print("="*60)
    print("TRULY PURE LEARNING TEST")
    print("="*60)
    
    for run in range(3):
        print(f"\n--- Run {run+1} ---")
        nav = TrulyPureSimple(maze)
        result = nav.navigate((size-2, size-2), max_steps=5000)
        
    return result


if __name__ == "__main__":
    test_simple()