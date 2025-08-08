#!/usr/bin/env python3
"""
Test TopK-based Pure Episodic Navigator
"""

import numpy as np
from typing import Dict, List, Tuple
import time
from pure_episodic_navigator import create_complex_maze, visualize_maze_with_path

class TopKEpisodicNavigator:
    """TopK-based efficient episodic navigator"""
    
    def __init__(self, maze: np.ndarray, k: int = 50, message_depth: int = 3):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.goal = (self.width-2, self.height-2)
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        
        # Parameters
        self.k = k  # Number of top episodes to consider
        self.message_depth = message_depth
        
        # Episode storage
        self.episodes = []
        
        # Visual memory
        self.visual_memory = {}
        self._update_visual_memory(1, 1)
        
        # Statistics
        self.wall_hits = 0
        self.moves = 0
        self.query_times = []
        
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
        
        embedding = np.array([
            pos[0] / self.width,
            pos[1] / self.height,
            {'up': -1.0, 'right': -0.33, 'down': 0.33, 'left': 1.0}[action],
            {'success': 1.0, 'wall': -1.0, 'visited': 0.0}[result],
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
    
    def get_topk_episodes(self, query_pos: Tuple[int, int]) -> List[Dict]:
        """Get top-k nearest episodes to query position"""
        if not self.episodes:
            return []
        
        start_time = time.time()
        
        # Calculate distances
        distances = []
        for ep in self.episodes:
            dist = abs(ep['pos'][0] - query_pos[0]) + abs(ep['pos'][1] - query_pos[1])
            distances.append((dist, ep))
        
        # Sort and get top-k
        distances.sort(key=lambda x: x[0])
        topk = [ep for _, ep in distances[:self.k]]
        
        # Always include goal episodes if any
        for ep in self.episodes:
            if ep['reached_goal'] and ep not in topk:
                topk.append(ep)
                if len(topk) > self.k:
                    # Remove furthest non-goal episode
                    for i in range(len(topk)-2, -1, -1):
                        if not topk[i]['reached_goal']:
                            topk.pop(i)
                            break
        
        query_time = time.time() - start_time
        self.query_times.append(query_time)
        
        return topk
    
    def simple_message_pass(self, episodes: List[Dict]) -> Dict[int, np.ndarray]:
        """Simple message passing on top-k episodes"""
        if not episodes:
            return {}
        
        # Initialize embeddings
        embeddings = {ep['id']: ep['embedding'].copy() for ep in episodes}
        
        # Message passing rounds
        for _ in range(self.message_depth):
            new_embeddings = {}
            
            for ep in episodes:
                # Find neighbors within topk
                neighbors = []
                for other in episodes:
                    if ep['id'] != other['id']:
                        dist = abs(ep['pos'][0] - other['pos'][0]) + \
                              abs(ep['pos'][1] - other['pos'][1])
                        if dist <= 2:
                            neighbors.append(other)
                
                if neighbors:
                    # Average neighbor embeddings
                    neighbor_embs = [embeddings[n['id']] for n in neighbors]
                    avg_neighbor = np.mean(neighbor_embs, axis=0)
                    
                    # Mix with different rates
                    new_emb = embeddings[ep['id']].copy()
                    new_emb[5] = 0.3 * embeddings[ep['id']][5] + 0.7 * avg_neighbor[5]  # Goal
                    new_emb[:5] = 0.7 * embeddings[ep['id']][:5] + 0.3 * avg_neighbor[:5]
                    
                    new_embeddings[ep['id']] = new_emb
                else:
                    new_embeddings[ep['id']] = embeddings[ep['id']]
            
            embeddings = new_embeddings
        
        return embeddings
    
    def evaluate_action(self, pos: Tuple[int, int], action: str) -> float:
        """Evaluate action using top-k episodes"""
        dx, dy = {'up': (0, -1), 'right': (1, 0), 
                 'down': (0, 1), 'left': (-1, 0)}[action]
        next_pos = (pos[0] + dx, pos[1] + dy)
        
        # Get top-k episodes
        topk_episodes = self.get_topk_episodes(next_pos)
        
        if not topk_episodes:
            return 0.0
        
        # Run message passing
        updated = self.simple_message_pass(topk_episodes)
        
        # Calculate score
        score = 0.0
        total_weight = 0.0
        
        for ep in topk_episodes:
            if ep['id'] in updated:
                # Extract signals
                goal_signal = updated[ep['id']][5]
                success_signal = updated[ep['id']][3]
                
                # Distance weight
                dist = abs(ep['pos'][0] - next_pos[0]) + abs(ep['pos'][1] - next_pos[1])
                weight = np.exp(-dist * 0.2)
                
                # Combined score
                ep_score = (goal_signal * 0.7 + success_signal * 0.3) * weight
                score += ep_score
                total_weight += weight
        
        if total_weight > 0:
            score /= total_weight
        
        # Exploration bonus
        if next_pos not in self.visited:
            score += 3.0
        
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
        
        # Mostly greedy
        if np.random.random() < 0.1:
            return np.random.choice(list(action_scores.keys()))
        else:
            return max(action_scores.keys(), key=lambda a: action_scores[a])
    
    def navigate(self, max_steps: int = 5000) -> Dict:
        """Navigate maze"""
        print(f"\nTopK Episodic Navigation (k={self.k})")
        print(f"Start: {self.position}, Goal: {self.goal}")
        print(f"Maze size: {self.width}×{self.height}")
        
        steps = 0
        start_time = time.time()
        
        while self.position != self.goal and steps < max_steps:
            if steps % 500 == 0 and steps > 0:
                dist = abs(self.position[0] - self.goal[0]) + abs(self.position[1] - self.goal[1])
                avg_query_time = np.mean(self.query_times[-100:]) * 1000  # ms
                print(f"Step {steps}: pos={self.position}, dist={dist}, "
                      f"episodes={len(self.episodes)}, avg_query={avg_query_time:.1f}ms")
            
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
        print(f"Avg query time: {np.mean(self.query_times)*1000:.2f}ms")
        
        return {
            'success': success,
            'steps': steps,
            'wall_hits': self.wall_hits,
            'time': elapsed,
            'path_length': len(self.path)
        }


def test_topk():
    """Test TopK navigator"""
    print("="*70)
    print("TOPK EPISODIC NAVIGATION TEST")
    print("="*70)
    
    # Test on different sizes
    for size in [15, 20, 25, 30]:
        print(f"\n{'='*50}")
        print(f"Testing {size}×{size} maze")
        print('='*50)
        
        maze = create_complex_maze(size, seed=42)
        
        # Try different k values
        k = min(100, size * size // 10)  # Adaptive k
        nav = TopKEpisodicNavigator(maze, k=k, message_depth=3)
        
        result = nav.navigate(max_steps=size * 150)
        
        if result['success']:
            efficiency = result['steps'] / (2 * (size - 2))
            print(f"✓ Efficiency: {efficiency:.2f}x optimal")
            
            # Save visualization
            import os
            os.makedirs('visualizations', exist_ok=True)
            visualize_maze_with_path(
                maze, nav.path,
                f'visualizations/topk_{size}x{size}.png'
            )
        else:
            print("✗ Failed")
            print(f"Explored {len(nav.visited)} cells")


if __name__ == "__main__":
    test_topk()