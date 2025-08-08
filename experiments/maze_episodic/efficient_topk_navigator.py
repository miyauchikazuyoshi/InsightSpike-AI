#!/usr/bin/env python3
"""
Efficient TopK Pure Episodic Navigator
=====================================

Maintains pure geDIG evaluation while being scalable.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
import time
from collections import defaultdict
import heapq

class EfficientTopKNavigator:
    """Efficient pure episodic navigator with TopK optimization"""
    
    def __init__(self, maze: np.ndarray, k: int = 50, message_depth: int = 3):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.goal = (self.width-2, self.height-2)
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        
        # Parameters
        self.k = k
        self.message_depth = message_depth
        
        # Episode storage with spatial indexing
        self.episodes = []
        self.spatial_index = defaultdict(list)  # Grid-based index
        self.goal_episodes = []  # Special storage for goal episodes
        
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
        """Create pure episode embedding"""
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
        """Add episode with spatial indexing"""
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
        
        # Spatial indexing (grid cells of size 5x5)
        grid_x, grid_y = pos[0] // 5, pos[1] // 5
        self.spatial_index[(grid_x, grid_y)].append(episode['id'])
        
        # Special handling for goal episodes
        if reached_goal:
            self.goal_episodes.append(episode['id'])
    
    def get_topk_neighbors(self, pos: Tuple[int, int], k: int = None) -> List[int]:
        """Get top-k nearest episodes using spatial index"""
        if k is None:
            k = self.k
        
        if not self.episodes:
            return []
        
        # Use heap for efficient top-k selection
        heap = []
        
        # Check spatial grid cells in expanding radius
        grid_x, grid_y = pos[0] // 5, pos[1] // 5
        checked = set()
        
        for radius in range(10):  # Expand search radius
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue  # Only check border cells
                    
                    grid_key = (grid_x + dx, grid_y + dy)
                    if grid_key in checked:
                        continue
                    checked.add(grid_key)
                    
                    for ep_id in self.spatial_index.get(grid_key, []):
                        ep = self.episodes[ep_id]
                        dist = abs(ep['pos'][0] - pos[0]) + abs(ep['pos'][1] - pos[1])
                        
                        if len(heap) < k:
                            heapq.heappush(heap, (-dist, ep_id))
                        elif dist < -heap[0][0]:
                            heapq.heapreplace(heap, (-dist, ep_id))
            
            if len(heap) >= k:
                break
        
        # Always include goal episodes
        for goal_id in self.goal_episodes:
            if goal_id not in [id for _, id in heap]:
                if len(heap) < k:
                    heapq.heappush(heap, (0, goal_id))  # Goal episodes have priority
        
        return [ep_id for _, ep_id in heap]
    
    def fast_message_pass(self, episode_ids: List[int]) -> Dict[int, np.ndarray]:
        """Fast message passing on selected episodes"""
        if not episode_ids:
            return {}
        
        # Build local adjacency for message passing
        episodes = [self.episodes[i] for i in episode_ids]
        embeddings = {ep['id']: ep['embedding'].copy() for ep in episodes}
        
        # Pre-compute neighbors
        neighbors = defaultdict(list)
        for i, ep1 in enumerate(episodes):
            for j, ep2 in enumerate(episodes):
                if i != j:
                    dist = abs(ep1['pos'][0] - ep2['pos'][0]) + \
                          abs(ep1['pos'][1] - ep2['pos'][1])
                    if dist <= 2:
                        neighbors[ep1['id']].append(ep2['id'])
        
        # Message passing rounds
        for _ in range(self.message_depth):
            new_embeddings = {}
            
            for ep_id in episode_ids:
                if ep_id not in neighbors or not neighbors[ep_id]:
                    new_embeddings[ep_id] = embeddings[ep_id]
                    continue
                
                # Aggregate neighbor messages
                neighbor_embs = [embeddings[n_id] for n_id in neighbors[ep_id]]
                avg_neighbor = np.mean(neighbor_embs, axis=0)
                
                # Different mixing rates for different dimensions
                new_emb = embeddings[ep_id].copy()
                new_emb[5] = 0.3 * embeddings[ep_id][5] + 0.7 * avg_neighbor[5]  # Goal
                new_emb[:5] = 0.7 * embeddings[ep_id][:5] + 0.3 * avg_neighbor[:5]
                
                new_embeddings[ep_id] = new_emb
            
            embeddings = new_embeddings
        
        return embeddings
    
    def evaluate_action_multihop(self, pos: Tuple[int, int], action: str) -> Tuple[float, int]:
        """Evaluate action using multi-hop with TopK"""
        dx, dy = {'up': (0, -1), 'right': (1, 0), 
                 'down': (0, 1), 'left': (-1, 0)}[action]
        next_pos = (pos[0] + dx, pos[1] + dy)
        
        # Try different hop counts with different k values
        best_score = -float('inf')
        best_hop = 1
        
        hop_configs = [(1, 20), (2, 30), (3, self.k)]  # (hops, k)
        
        for n_hops, k in hop_configs:
            # Get top-k episodes
            episode_ids = self.get_topk_neighbors(next_pos, k)
            
            if not episode_ids:
                continue
            
            # Run message passing
            updated = self.fast_message_pass(episode_ids)
            
            # Calculate score
            score = 0.0
            total_weight = 0.0
            
            for ep_id in episode_ids:
                if ep_id in updated:
                    ep = self.episodes[ep_id]
                    
                    # Extract signals
                    goal_signal = updated[ep_id][5]
                    success_signal = updated[ep_id][3]
                    
                    # Distance weight
                    dist = abs(ep['pos'][0] - next_pos[0]) + abs(ep['pos'][1] - next_pos[1])
                    weight = np.exp(-dist * 0.2)
                    
                    # Combined score
                    ep_score = (goal_signal * 0.6 + success_signal * 0.4) * weight
                    score += ep_score
                    total_weight += weight
            
            if total_weight > 0:
                score /= total_weight
            
            # Small bonus for larger hop count
            score += n_hops * 0.05
            
            if score > best_score:
                best_score = score
                best_hop = n_hops
        
        self.hop_selections[f'{best_hop}-hop'] += 1
        return best_score, best_hop
    
    def decide_action(self) -> str:
        """Decide action using efficient evaluation"""
        visual = self.visual_memory.get(self.position, {})
        
        action_scores = {}
        
        for action in ['up', 'right', 'down', 'left']:
            if visual.get(action) == 'wall':
                action_scores[action] = -10.0
                continue
            
            score, _ = self.evaluate_action_multihop(self.position, action)
            action_scores[action] = score
            
            # Exploration bonus
            dx, dy = {'up': (0, -1), 'right': (1, 0), 
                     'down': (0, 1), 'left': (-1, 0)}[action]
            next_pos = (self.position[0] + dx, self.position[1] + dy)
            
            if next_pos not in self.visited:
                action_scores[action] += 2.0
        
        if not action_scores:
            return 'up'
        
        # Softmax selection
        actions = list(action_scores.keys())
        values = np.array(list(action_scores.values()))
        
        temperature = 0.3
        exp_values = np.exp((values - values.max()) / temperature)
        probs = exp_values / exp_values.sum()
        
        return np.random.choice(actions, p=probs)
    
    def navigate(self, max_steps: int = 5000) -> Dict:
        """Navigate maze efficiently"""
        print(f"\nEfficient TopK Episodic Navigation (k={self.k})")
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
                
                # Show hop distribution
                total = sum(self.hop_selections.values())
                if total > 0:
                    print("  Hop usage:", end=" ")
                    for hop, count in self.hop_selections.items():
                        print(f"{hop}: {count/total*100:.1f}%", end=" ")
                    print()
            
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
        
        # Final hop distribution
        print(f"\nFinal hop selection distribution:")
        total = sum(self.hop_selections.values())
        if total > 0:
            for hop, count in self.hop_selections.items():
                print(f"  {hop}: {count} ({count/total*100:.1f}%)")
        
        return {
            'success': success,
            'steps': steps,
            'wall_hits': self.wall_hits,
            'time': elapsed,
            'path_length': len(self.path),
            'hop_selections': dict(self.hop_selections)
        }


def test_efficient():
    """Test efficient TopK navigator"""
    from pure_episodic_navigator import create_complex_maze, visualize_maze_with_path
    import os
    
    print("="*70)
    print("EFFICIENT TOPK EPISODIC NAVIGATION TEST")
    print("="*70)
    
    # Test on multiple sizes
    for size in [15, 20, 25, 30]:
        print(f"\n{'='*50}")
        print(f"Testing {size}×{size} maze")
        print('='*50)
        
        maze = create_complex_maze(size, seed=42)
        
        # Adaptive k based on maze size
        k = min(50, max(30, size * 2))
        nav = EfficientTopKNavigator(maze, k=k, message_depth=3)
        
        result = nav.navigate(max_steps=size * size * 5)
        
        if result['success']:
            efficiency = result['steps'] / (2 * (size - 2))
            print(f"✓ Efficiency: {efficiency:.2f}x optimal")
            
            # Save visualization
            os.makedirs('visualizations', exist_ok=True)
            visualize_maze_with_path(
                maze, nav.path,
                f'visualizations/efficient_topk_{size}x{size}.png'
            )
        else:
            print("✗ Failed")
            print(f"Explored {len(nav.visited)} cells")


if __name__ == "__main__":
    test_efficient()