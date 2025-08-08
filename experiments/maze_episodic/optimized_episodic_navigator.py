#!/usr/bin/env python3
"""
Optimized Pure Episodic Navigator
=================================

Efficient implementation for larger mazes.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
import time
from collections import deque
import heapq

class OptimizedEpisodicNavigator:
    """Optimized pure episodic navigator without visit counts"""
    
    def __init__(self, maze: np.ndarray):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.goal = (self.width-2, self.height-2)
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        
        # Efficient episode storage
        self.episode_buffer = deque(maxlen=500)  # Limit memory
        self.spatial_index = {}  # Grid-based indexing for fast lookup
        self.goal_episodes = []  # Special storage for goal episodes
        
        # Visual memory
        self.visual_memory = {}
        self._update_visual_memory(1, 1)
        
        # Statistics
        self.wall_hits = 0
        self.moves = 0
        self.episode_count = 0
        
    def _update_visual_memory(self, x: int, y: int):
        """Update visual memory"""
        self.visual_memory[(x, y)] = {}
        for action, (dx, dy) in {'up': (0, -1), 'right': (1, 0), 
                                'down': (0, 1), 'left': (-1, 0)}.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                self.visual_memory[(x, y)][action] = 'path' if self.maze[ny, nx] == 0 else 'wall'
    
    def create_embedding(self, pos: Tuple[int, int], action: str, 
                        result: str, reached_goal: bool) -> np.ndarray:
        """Create pure episode embedding"""
        visual = self.visual_memory.get(pos, {})
        wall_count = sum(1 for d in ['up', 'right', 'down', 'left']
                        if visual.get(d) == 'wall')
        
        return np.array([
            pos[0] / self.width,
            pos[1] / self.height,
            {'up': -1.0, 'right': -0.33, 'down': 0.33, 'left': 1.0}[action],
            {'success': 1.0, 'wall': -1.0, 'visited': 0.0}[result],
            (wall_count - 2) / 2,
            10.0 if reached_goal else 0.0
        ])
    
    def add_episode(self, pos: Tuple[int, int], action: str, 
                   result: str, reached_goal: bool):
        """Add episode with spatial indexing"""
        embedding = self.create_embedding(pos, action, result, reached_goal)
        
        episode = {
            'id': self.episode_count,
            'pos': pos,
            'action': action,
            'result': result,
            'reached_goal': reached_goal,
            'embedding': embedding
        }
        
        self.episode_count += 1
        self.episode_buffer.append(episode)
        
        # Spatial indexing (5x5 grid cells)
        grid_x, grid_y = pos[0] // 5, pos[1] // 5
        grid_key = (grid_x, grid_y)
        
        if grid_key not in self.spatial_index:
            self.spatial_index[grid_key] = []
        self.spatial_index[grid_key].append(episode)
        
        # Keep only recent episodes per grid cell
        if len(self.spatial_index[grid_key]) > 20:
            self.spatial_index[grid_key] = self.spatial_index[grid_key][-20:]
        
        # Special handling for goal episodes
        if reached_goal:
            self.goal_episodes.append(episode)
    
    def get_relevant_episodes(self, pos: Tuple[int, int], radius: int = 2) -> List[Dict]:
        """Get spatially relevant episodes"""
        episodes = []
        grid_x, grid_y = pos[0] // 5, pos[1] // 5
        
        # Check nearby grid cells
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                grid_key = (grid_x + dx, grid_y + dy)
                if grid_key in self.spatial_index:
                    episodes.extend(self.spatial_index[grid_key])
        
        # Always include goal episodes
        episodes.extend(self.goal_episodes)
        
        # Remove duplicates
        seen = set()
        unique_episodes = []
        for ep in episodes:
            if ep['id'] not in seen:
                seen.add(ep['id'])
                unique_episodes.append(ep)
        
        return unique_episodes
    
    def simple_message_pass(self, episodes: List[Dict]) -> Dict[int, np.ndarray]:
        """Simplified message passing"""
        if not episodes:
            return {}
        
        # Group by proximity
        embeddings = {}
        for ep in episodes:
            embeddings[ep['id']] = ep['embedding'].copy()
        
        # Single round of averaging with neighbors
        new_embeddings = {}
        for ep in episodes:
            neighbors = []
            for other in episodes:
                if ep['id'] != other['id']:
                    dist = abs(ep['pos'][0] - other['pos'][0]) + \
                          abs(ep['pos'][1] - other['pos'][1])
                    if dist <= 2:
                        neighbors.append(other['embedding'])
            
            if neighbors:
                avg_neighbor = np.mean(neighbors, axis=0)
                # Stronger goal signal propagation
                new_emb = embeddings[ep['id']].copy()
                new_emb[5] = 0.3 * embeddings[ep['id']][5] + 0.7 * avg_neighbor[5]
                new_emb[:5] = 0.8 * embeddings[ep['id']][:5] + 0.2 * avg_neighbor[:5]
                new_embeddings[ep['id']] = new_emb
            else:
                new_embeddings[ep['id']] = embeddings[ep['id']]
        
        return new_embeddings
    
    def evaluate_action(self, pos: Tuple[int, int], action: str) -> float:
        """Evaluate action using episodic memory"""
        # Get next position
        dx, dy = {'up': (0, -1), 'right': (1, 0), 
                 'down': (0, 1), 'left': (-1, 0)}[action]
        next_pos = (pos[0] + dx, pos[1] + dy)
        
        # Get relevant episodes
        episodes = self.get_relevant_episodes(next_pos)
        
        if not episodes:
            return 0.0
        
        # Simple message passing
        updated = self.simple_message_pass(episodes)
        
        # Calculate score
        score = 0.0
        total_weight = 0.0
        
        for ep in episodes:
            if ep['id'] in updated:
                # Extract signals
                goal_signal = updated[ep['id']][5]
                success_signal = updated[ep['id']][3]
                
                # Distance weight
                dist = abs(ep['pos'][0] - next_pos[0]) + \
                      abs(ep['pos'][1] - next_pos[1])
                weight = np.exp(-dist * 0.3)
                
                # Combined score
                ep_score = (goal_signal * 0.7 + success_signal * 0.3) * weight
                score += ep_score
                total_weight += weight
        
        if total_weight > 0:
            score /= total_weight
        
        return score
    
    def decide_action(self) -> str:
        """Fast action decision"""
        visual = self.visual_memory.get(self.position, {})
        
        action_scores = {}
        
        for action in ['up', 'right', 'down', 'left']:
            # Skip walls
            if visual.get(action) == 'wall':
                continue
            
            # Evaluate
            score = self.evaluate_action(self.position, action)
            
            # Exploration bonus
            dx, dy = {'up': (0, -1), 'right': (1, 0), 
                     'down': (0, 1), 'left': (-1, 0)}[action]
            next_pos = (self.position[0] + dx, self.position[1] + dy)
            
            if next_pos not in self.visited:
                score += 3.0
            
            action_scores[action] = score
        
        if not action_scores:
            # All walls, try any direction
            return np.random.choice(['up', 'right', 'down', 'left'])
        
        # Greedy selection with small randomness
        if np.random.random() < 0.1:
            return np.random.choice(list(action_scores.keys()))
        else:
            return max(action_scores.keys(), key=lambda a: action_scores[a])
    
    def navigate(self, max_steps: int = 5000) -> Dict:
        """Navigate maze efficiently"""
        print(f"\nOptimized Episodic Navigation")
        print(f"Start: {self.position}, Goal: {self.goal}")
        print(f"Maze size: {self.width}×{self.height}")
        
        steps = 0
        start_time = time.time()
        
        while self.position != self.goal and steps < max_steps:
            if steps % 500 == 0 and steps > 0:
                dist = abs(self.position[0] - self.goal[0]) + \
                      abs(self.position[1] - self.goal[1])
                print(f"Step {steps}: pos={self.position}, dist={dist}, "
                      f"episodes={self.episode_count}")
            
            # Decide action
            action = self.decide_action()
            
            # Execute
            old_pos = self.position
            dx, dy = {'up': (0, -1), 'right': (1, 0), 
                     'down': (0, 1), 'left': (-1, 0)}[action]
            new_pos = (self.position[0] + dx, self.position[1] + dy)
            
            result = 'wall'
            reached_goal = False
            
            if (0 <= new_pos[0] < self.width and 
                0 <= new_pos[1] < self.height and
                self.maze[new_pos[1], new_pos[0]] == 0):
                
                if new_pos in self.visited:
                    result = 'visited'
                else:
                    result = 'success'
                
                self.position = new_pos
                self.visited.add(new_pos)
                self.path.append(new_pos)
                self.moves += 1
                self._update_visual_memory(new_pos[0], new_pos[1])
                
                if new_pos == self.goal:
                    reached_goal = True
            else:
                self.wall_hits += 1
            
            # Add episode
            self.add_episode(old_pos, action, result, reached_goal)
            steps += 1
        
        elapsed = time.time() - start_time
        success = self.position == self.goal
        
        print(f"\nComplete! Success: {success}")
        print(f"Steps: {steps}, Wall hits: {self.wall_hits}")
        print(f"Time: {elapsed:.2f}s")
        
        if self.goal_episodes:
            print(f"Goal reached at step: {self.goal_episodes[0]['id']}")
        
        return {
            'success': success,
            'steps': steps,
            'wall_hits': self.wall_hits,
            'time': elapsed,
            'path_length': len(self.path)
        }


def test_optimized():
    """Test optimized navigator"""
    from pure_episodic_navigator import create_complex_maze, visualize_maze_with_path
    
    print("="*60)
    print("OPTIMIZED EPISODIC NAVIGATION TEST")
    print("="*60)
    
    # Test multiple sizes
    for size in [25, 30, 40]:
        print(f"\n{'='*50}")
        print(f"Testing {size}×{size} maze")
        print('='*50)
        
        maze = create_complex_maze(size, seed=42)
        nav = OptimizedEpisodicNavigator(maze)
        
        # Increase max_steps based on size
        max_steps = size * size * 10  # Much larger timeout
        result = nav.navigate(max_steps=max_steps)
        
        if result['success']:
            efficiency = result['steps'] / (2 * (size - 2))
            print(f"✓ Efficiency: {efficiency:.2f}x optimal")
            
            # Save visualization
            import os
            os.makedirs('visualizations', exist_ok=True)
            visualize_maze_with_path(
                maze, nav.path,
                f'visualizations/optimized_{size}x{size}.png'
            )
        else:
            print("✗ Failed")
            break


if __name__ == "__main__":
    test_optimized()