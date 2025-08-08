#!/usr/bin/env python3
"""
Pure Episodic Navigator V2
==========================

Improved version without visit counts, using multi-hop evaluation.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

class PureEpisodicNavigatorV2:
    """Improved pure episodic navigator"""
    
    def __init__(self, maze: np.ndarray, message_depth: int = 3):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.goal = (self.width-2, self.height-2)
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        
        # Episode storage
        self.episodes = []
        self.message_depth = message_depth
        
        # Visual memory
        self.visual_memory = {}
        self._update_visual_memory(1, 1)
        
        # Statistics
        self.wall_hits = 0
        self.moves = 0
        self.hop_selections = {'1-hop': 0, '2-hop': 0, '3-hop': 0}
        
    def _update_visual_memory(self, x: int, y: int):
        """Update visual memory"""
        self.visual_memory[(x, y)] = {}
        for action, (dx, dy) in {'up': (0, -1), 'right': (1, 0), 
                                'down': (0, 1), 'left': (-1, 0)}.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                self.visual_memory[(x, y)][action] = 'path' if self.maze[ny, nx] == 0 else 'wall'
    
    def create_episode_embedding(self, pos: Tuple[int, int], action: str, 
                               result: str, reached_goal: bool) -> np.ndarray:
        """Create episode embedding"""
        visual = self.visual_memory.get(pos, {})
        wall_count = sum(1 for d in ['up', 'right', 'down', 'left']
                        if visual.get(d) == 'wall')
        
        # Simpler embedding
        embedding = np.array([
            pos[0] / self.width,
            pos[1] / self.height,
            {'up': -1.0, 'right': -0.33, 'down': 0.33, 'left': 1.0}[action],
            1.0 if result == 'success' else -1.0,
            (wall_count - 2) / 2,
            10.0 if reached_goal else 0.0
        ])
        
        return embedding
    
    def add_episode(self, pos: Tuple[int, int], action: str, 
                   result: str, reached_goal: bool):
        """Add episode"""
        embedding = self.create_episode_embedding(pos, action, result, reached_goal)
        
        episode = {
            'id': len(self.episodes),
            'pos': pos,
            'action': action,
            'result': result,
            'reached_goal': reached_goal,
            'embedding': embedding
        }
        
        self.episodes.append(episode)
    
    def find_relevant_episodes(self, pos: Tuple[int, int], radius: int = 10) -> List[Dict]:
        """Find episodes near a position"""
        relevant = []
        
        for ep in self.episodes:
            dist = abs(ep['pos'][0] - pos[0]) + abs(ep['pos'][1] - pos[1])
            if dist <= radius:
                relevant.append(ep)
        
        # Always include goal episodes
        for ep in self.episodes:
            if ep['reached_goal'] and ep not in relevant:
                relevant.append(ep)
        
        return relevant
    
    def propagate_goal_signal(self, episodes: List[Dict]) -> Dict[int, float]:
        """Propagate goal signal through episodes"""
        # Initialize signals
        signals = {}
        for ep in episodes:
            signals[ep['id']] = ep['embedding'][5]  # Goal signal
        
        # Propagate for multiple rounds
        for _ in range(self.message_depth):
            new_signals = signals.copy()
            
            for ep in episodes:
                if ep['id'] not in signals:
                    continue
                
                # Find neighbors
                neighbors = []
                for other in episodes:
                    if other['id'] != ep['id']:
                        dist = abs(ep['pos'][0] - other['pos'][0]) + \
                              abs(ep['pos'][1] - other['pos'][1])
                        if dist <= 2:
                            neighbors.append(other['id'])
                
                # Update signal
                if neighbors:
                    neighbor_signals = [signals.get(n, 0) for n in neighbors]
                    avg_neighbor = np.mean(neighbor_signals)
                    # Strong propagation
                    new_signals[ep['id']] = 0.3 * signals[ep['id']] + 0.7 * avg_neighbor
            
            signals = new_signals
        
        return signals
    
    def evaluate_action(self, pos: Tuple[int, int], action: str) -> float:
        """Evaluate action using episodic memory"""
        dx, dy = {'up': (0, -1), 'right': (1, 0), 
                 'down': (0, 1), 'left': (-1, 0)}[action]
        next_pos = (pos[0] + dx, pos[1] + dy)
        
        # Get relevant episodes
        episodes = self.find_relevant_episodes(next_pos)
        
        if not episodes:
            return 0.0
        
        # Propagate goal signals
        goal_signals = self.propagate_goal_signal(episodes)
        
        # Calculate score
        score = 0.0
        total_weight = 0.0
        
        for ep in episodes:
            # Goal signal strength
            goal_signal = goal_signals.get(ep['id'], 0)
            
            # Success/failure from episode
            success_signal = 1.0 if ep['result'] == 'success' else -1.0
            
            # Distance weight
            dist = abs(ep['pos'][0] - next_pos[0]) + abs(ep['pos'][1] - next_pos[1])
            weight = np.exp(-dist * 0.1)
            
            # Combined score
            ep_score = (goal_signal * 0.7 + success_signal * 0.3) * weight
            score += ep_score
            total_weight += weight
        
        if total_weight > 0:
            score /= total_weight
        
        # Strong exploration bonus
        if next_pos not in self.visited:
            score += 5.0
        
        # Penalty for excessive revisits
        revisit_count = sum(1 for ep in self.episodes if ep['pos'] == next_pos)
        if revisit_count > 3:
            score -= (revisit_count - 3) * 0.5
        
        return score
    
    def decide_action(self) -> str:
        """Decide action"""
        visual = self.visual_memory.get(self.position, {})
        
        action_scores = {}
        
        for action in ['up', 'right', 'down', 'left']:
            if visual.get(action) == 'wall':
                continue
            
            score = self.evaluate_action(self.position, action)
            action_scores[action] = score
        
        if not action_scores:
            return np.random.choice(['up', 'right', 'down', 'left'])
        
        # Mostly greedy with small exploration
        if np.random.random() < 0.1:
            return np.random.choice(list(action_scores.keys()))
        else:
            return max(action_scores.keys(), key=lambda a: action_scores[a])
    
    def navigate(self, max_steps: int = 5000) -> Dict:
        """Navigate maze"""
        print(f"\nPure Episodic Navigation V2")
        print(f"Start: {self.position}, Goal: {self.goal}")
        print(f"Maze size: {self.width}×{self.height}")
        
        steps = 0
        start_time = time.time()
        
        while self.position != self.goal and steps < max_steps:
            if steps % 500 == 0 and steps > 0:
                dist = abs(self.position[0] - self.goal[0]) + abs(self.position[1] - self.goal[1])
                coverage = len(self.visited) / (self.width * self.height) * 100
                print(f"Step {steps}: pos={self.position}, dist={dist}, "
                      f"coverage={coverage:.1f}%, episodes={len(self.episodes)}")
            
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
        
        # Find when goal was first reached
        goal_episodes = [ep for ep in self.episodes if ep['reached_goal']]
        if goal_episodes:
            print(f"Goal reached at step: {goal_episodes[0]['id']}")
        
        return {
            'success': success,
            'steps': steps,
            'wall_hits': self.wall_hits,
            'time': elapsed,
            'path_length': len(self.path)
        }


def test_v2():
    """Test improved version"""
    from pure_episodic_navigator import create_complex_maze, visualize_maze_with_path
    
    print("="*70)
    print("PURE EPISODIC NAVIGATION V2 TEST")
    print("="*70)
    
    # Test on multiple sizes
    for size in [15, 20, 25, 30]:
        print(f"\n{'='*50}")
        print(f"Testing {size}×{size} maze")
        print('='*50)
        
        maze = create_complex_maze(size, seed=42)
        nav = PureEpisodicNavigatorV2(maze, message_depth=3)
        
        result = nav.navigate(max_steps=size * size * 10)
        
        if result['success']:
            efficiency = result['steps'] / (2 * (size - 2))
            print(f"✓ Efficiency: {efficiency:.2f}x optimal")
            
            # Save visualization
            os.makedirs('visualizations', exist_ok=True)
            visualize_maze_with_path(
                maze, nav.path,
                f'visualizations/pure_v2_{size}x{size}.png'
            )
        else:
            print("✗ Failed")
            print(f"Explored {len(nav.visited)} cells")
            break


if __name__ == "__main__":
    test_v2()