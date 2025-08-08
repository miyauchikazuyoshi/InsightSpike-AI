#!/usr/bin/env python3
"""
Pure Memory-Based Navigator
- Stores movement episodes: (x, y, direction, success, wall/path, visit_count, goal)  
- No bonuses or penalties - pure memory-based decision making
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional


class PureMemoryNavigator:
    """Navigator using pure episodic memory"""
    
    def __init__(self, maze: np.ndarray):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = self._find_start()
        self.goal = self._find_goal()
        
        # Episode memory
        self.episodes = []  # List of movement episodes
        self.visit_counts = {}  # Visit count per position
        
        # Path tracking
        self.path = [self.position]
        self.wall_hits = 0
        
        # Action mapping
        self.actions = ['up', 'right', 'down', 'left']
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        self.action_deltas = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
    
    def _find_start(self) -> Tuple[int, int]:
        for i in range(self.height):
            for j in range(self.width):
                if self.maze[i, j] == 0:
                    return (i, j)
        return (1, 1)
    
    def _find_goal(self) -> Tuple[int, int]:
        for i in range(self.height-1, -1, -1):
            for j in range(self.width-1, -1, -1):
                if self.maze[i, j] == 0:
                    return (i, j)
        return (self.height-2, self.width-2)
    
    def _update_visit_count(self):
        """Update visit count for current position"""
        pos = self.position
        if pos not in self.visit_counts:
            self.visit_counts[pos] = 0
        self.visit_counts[pos] += 1
    
    def _observe_directions(self) -> Dict[str, bool]:
        """Observe which directions have walls/paths"""
        observations = {}
        x, y = self.position
        
        for action, (dx, dy) in self.action_deltas.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.height and 0 <= ny < self.width:
                observations[action] = (self.maze[nx, ny] == 0)  # True if path
            else:
                observations[action] = False  # Out of bounds = wall
        
        return observations
    
    def _create_episode(self, action: str, success: bool, is_path: bool) -> Dict:
        """Create a movement episode"""
        x, y = self.position
        episode = {
            'x': x,
            'y': y,
            'direction': action,
            'success': success,
            'is_path': is_path,
            'visit_count': self.visit_counts.get((x, y), 1),
            'is_goal': (x, y) == self.goal,
            'timestamp': len(self.episodes)  # For recency
        }
        return episode
    
    def _search_memory(self) -> Dict[str, float]:
        """Search episodic memory for relevant experiences"""
        x, y = self.position
        action_scores = {action: 0.0 for action in self.actions}
        
        # Look for episodes from current position
        relevant_episodes = [ep for ep in self.episodes if ep['x'] == x and ep['y'] == y]
        
        if not relevant_episodes:
            # No memory from this position - explore randomly
            return {action: 0.25 for action in self.actions}
        
        # Aggregate successful experiences
        for episode in relevant_episodes:
            action = episode['direction']
            
            # Pure memory: successful moves get positive score
            if episode['success']:
                # More recent episodes have slightly more weight
                recency_weight = 1.0 + (episode['timestamp'] / len(self.episodes)) * 0.1
                action_scores[action] += recency_weight
            else:
                # Failed moves get negative score
                action_scores[action] -= 0.5
        
        # Normalize to probabilities (avoid negative)
        min_score = min(action_scores.values())
        if min_score < 0:
            for action in action_scores:
                action_scores[action] -= min_score
        
        total = sum(action_scores.values())
        if total > 0:
            for action in action_scores:
                action_scores[action] /= total
        else:
            # If all scores are 0, uniform random
            action_scores = {action: 0.25 for action in self.actions}
        
        return action_scores
    
    def get_action(self) -> str:
        """Get action based on pure memory"""
        self._update_visit_count()
        
        # Get observations (for creating episodes)
        observations = self._observe_directions()
        
        # Search memory for action scores
        action_scores = self._search_memory()
        
        # Convert to probability distribution
        actions = list(action_scores.keys())
        probs = list(action_scores.values())
        
        # Sample action
        return np.random.choice(actions, p=probs)
    
    def move(self, action: str) -> bool:
        """Execute action and store episode"""
        if action not in self.actions:
            return False
        
        x, y = self.position
        dx, dy = self.action_deltas[action]
        new_x, new_y = x + dx, y + dy
        
        # Check if move is valid
        observations = self._observe_directions()
        is_path = observations[action]
        
        success = False
        if is_path and 0 <= new_x < self.height and 0 <= new_y < self.width:
            if self.maze[new_x, new_y] == 0:
                self.position = (new_x, new_y)
                self.path.append(self.position)
                success = True
        
        if not success:
            self.wall_hits += 1
        
        # Store episode
        episode = self._create_episode(action, success, is_path)
        self.episodes.append(episode)
        
        return success
    
    def navigate(self, max_steps: int = 5000) -> Dict:
        """Navigate using pure memory"""
        start_time = time.time()
        
        for step in range(max_steps):
            if self.position == self.goal:
                total_time = time.time() - start_time
                
                print(f"\n🎉 SUCCESS with pure memory!")
                print(f"  Steps: {step}")
                print(f"  Episodes: {len(self.episodes)}")
                print(f"  Wall hits: {self.wall_hits}")
                
                return {
                    'success': True,
                    'steps': step,
                    'episodes': len(self.episodes),
                    'wall_hits': self.wall_hits,
                    'path': self.path,
                    'visit_counts': self.visit_counts,
                    'total_time': total_time
                }
            
            # Get and execute action
            action = self.get_action()
            self.move(action)
            
            # Progress report
            if step % 100 == 0 and step > 0:
                dist = abs(self.position[0]-self.goal[0]) + abs(self.position[1]-self.goal[1])
                hit_rate = self.wall_hits / step * 100
                max_visits = max(self.visit_counts.values()) if self.visit_counts else 0
                
                print(f"Step {step}: pos={self.position}, dist={dist}, "
                      f"wall_hits={self.wall_hits} ({hit_rate:.1f}%), "
                      f"max_visits={max_visits}")
        
        total_time = time.time() - start_time
        
        return {
            'success': False,
            'steps': max_steps,
            'episodes': len(self.episodes),
            'wall_hits': self.wall_hits,
            'path': self.path,
            'visit_counts': self.visit_counts,
            'total_time': total_time
        }


def test_pure_memory():
    """Test pure memory navigator"""
    # Simple maze
    maze = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ])
    
    print("Pure Memory Navigation Test")
    print("Maze (0=path, 1=wall):")
    for row in maze:
        print(' '.join(['.' if x == 0 else '#' for x in row]))
    print(f"Start: (0,0), Goal: (4,4)")
    print("-" * 40)
    
    navigator = PureMemoryNavigator(maze)
    result = navigator.navigate(max_steps=1000)
    
    if result['success']:
        print(f"\n✅ Solved in {result['steps']} steps!")
        print(f"Path length: {len(result['path'])}")
        
        # Show most visited positions
        sorted_visits = sorted(result['visit_counts'].items(), 
                             key=lambda x: x[1], reverse=True)[:3]
        print(f"\nMost visited positions:")
        for pos, count in sorted_visits:
            print(f"  {pos}: {count} visits")
    else:
        print(f"\n❌ Failed after {result['steps']} steps")
    
    return result


if __name__ == "__main__":
    test_pure_memory()