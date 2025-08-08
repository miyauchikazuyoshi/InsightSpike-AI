#!/usr/bin/env python3
"""
Pure Episodic Navigator
=======================

Navigator using pure episodic memory without visit counts.
Multi-hop evaluation and deep message passing for efficient navigation.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

class PureEpisodicNavigator:
    """Pure episodic memory navigator without visit counts"""
    
    def __init__(self, maze: np.ndarray, message_depth: int = 3):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.goal = (self.width-2, self.height-2)
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        
        # Episode storage
        self.episodes = []
        self.episode_graph = {}  # adjacency for message passing
        self.message_depth = message_depth
        
        # Visual memory (local observation)
        self.visual_memory = {}
        self._update_visual_memory(1, 1)
        
        # Statistics
        self.wall_hits = 0
        self.moves = 0
        self.hop_selections = {'1-hop': 0, '2-hop': 0, '3-hop': 0}
        
    def _update_visual_memory(self, x: int, y: int):
        """Update visual memory from current position"""
        self.visual_memory[(x, y)] = {}
        for action, (dx, dy) in {'up': (0, -1), 'right': (1, 0), 
                                'down': (0, 1), 'left': (-1, 0)}.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                self.visual_memory[(x, y)][action] = 'path' if self.maze[ny, nx] == 0 else 'wall'
    
    def create_episode_embedding(self, pos: Tuple[int, int], action: str, 
                               result: str, reached_goal: bool) -> np.ndarray:
        """Create pure episode embedding without visit count"""
        visual = self.visual_memory.get(pos, {})
        
        # Count visible walls (local topology)
        wall_count = sum(1 for d in ['up', 'right', 'down', 'left']
                        if visual.get(d) == 'wall')
        
        # Detect junction (3+ open paths)
        open_paths = 4 - wall_count
        is_junction = open_paths >= 3
        
        # 6-dimensional embedding
        embedding = np.array([
            pos[0] / self.width,  # Position X [0, 1]
            pos[1] / self.height,  # Position Y [0, 1]
            {'up': -1.0, 'right': -0.33, 'down': 0.33, 'left': 1.0}[action],  # Action
            {'success': 1.0, 'wall': -1.0, 'visited': 0.0}[result],  # Result
            (wall_count - 2) / 2,  # Local topology [-1, 1]
            10.0 if reached_goal else 0.0  # Goal signal
        ])
        
        return embedding
    
    def add_episode(self, pos: Tuple[int, int], action: str, 
                   result: str, reached_goal: bool):
        """Add episode and update graph"""
        embedding = self.create_episode_embedding(pos, action, result, reached_goal)
        
        episode = {
            'id': len(self.episodes),
            'pos': pos,
            'action': action,
            'result': result,
            'reached_goal': reached_goal,
            'embedding': embedding
        }
        
        # Add to graph and connect to nearby episodes
        self.episode_graph[episode['id']] = []
        
        # Connect to spatially close episodes
        for other in self.episodes:
            dist = abs(pos[0] - other['pos'][0]) + abs(pos[1] - other['pos'][1])
            if dist <= 3:  # Within 3 Manhattan distance
                self.episode_graph[episode['id']].append(other['id'])
                if other['id'] in self.episode_graph:
                    self.episode_graph[other['id']].append(episode['id'])
        
        self.episodes.append(episode)
    
    def get_n_hop_episodes(self, start_pos: Tuple[int, int], n_hops: int) -> List[int]:
        """Get episodes within n hops from a position"""
        if not self.episodes:
            return []
        
        # Find closest episode to start position
        min_dist = float('inf')
        start_ep_id = None
        
        for ep in self.episodes:
            dist = abs(ep['pos'][0] - start_pos[0]) + abs(ep['pos'][1] - start_pos[1])
            if dist < min_dist:
                min_dist = dist
                start_ep_id = ep['id']
        
        if start_ep_id is None:
            return []
        
        # BFS to find n-hop neighbors
        visited = {start_ep_id}
        current = {start_ep_id}
        
        for _ in range(n_hops):
            next_layer = set()
            for ep_id in current:
                if ep_id in self.episode_graph:
                    for neighbor_id in self.episode_graph[ep_id]:
                        if neighbor_id not in visited:
                            next_layer.add(neighbor_id)
                            visited.add(neighbor_id)
            current = next_layer
            if not current:
                break
        
        return list(visited)
    
    def deep_message_pass(self, episode_ids: List[int]) -> Dict[int, np.ndarray]:
        """Deep message passing on episode subgraph"""
        if not episode_ids:
            return {}
        
        # Initialize embeddings
        embeddings = {ep_id: self.episodes[ep_id]['embedding'].copy() 
                     for ep_id in episode_ids}
        
        # Multiple rounds of message passing
        for _ in range(self.message_depth):
            new_embeddings = {}
            
            for ep_id in episode_ids:
                if ep_id not in self.episode_graph:
                    new_embeddings[ep_id] = embeddings[ep_id]
                    continue
                
                # Collect messages from neighbors
                messages = []
                for neighbor_id in self.episode_graph[ep_id]:
                    if neighbor_id in episode_ids and neighbor_id in embeddings:
                        messages.append(embeddings[neighbor_id])
                
                if messages:
                    # Average neighbors with different weights for different dimensions
                    avg_msg = np.mean(messages, axis=0)
                    new_emb = embeddings[ep_id].copy()
                    
                    # Different mixing rates
                    for i in range(len(new_emb)):
                        if i == 5:  # Goal dimension - stronger propagation
                            new_emb[i] = 0.4 * embeddings[ep_id][i] + 0.6 * avg_msg[i]
                        else:
                            new_emb[i] = 0.7 * embeddings[ep_id][i] + 0.3 * avg_msg[i]
                    
                    new_embeddings[ep_id] = new_emb
                else:
                    new_embeddings[ep_id] = embeddings[ep_id]
            
            embeddings = new_embeddings
        
        return embeddings
    
    def evaluate_action_multihop(self, pos: Tuple[int, int], action: str) -> Tuple[float, int]:
        """Evaluate action using multi-hop consideration"""
        # Get next position
        dx, dy = {'up': (0, -1), 'right': (1, 0), 
                 'down': (0, 1), 'left': (-1, 0)}[action]
        next_pos = (pos[0] + dx, pos[1] + dy)
        
        # Try different hop counts
        best_score = -float('inf')
        best_hop = 1
        
        for n_hops in [1, 2, 3]:
            episode_ids = self.get_n_hop_episodes(next_pos, n_hops)
            
            if not episode_ids:
                continue
            
            # Run message passing
            updated = self.deep_message_pass(episode_ids)
            
            # Calculate score based on updated embeddings
            score = 0.0
            for ep_id in episode_ids:
                if ep_id in updated:
                    ep = self.episodes[ep_id]
                    
                    # Extract signals from embedding
                    goal_signal = updated[ep_id][5]  # Goal dimension
                    success_signal = updated[ep_id][3]  # Result dimension
                    
                    # Distance weight
                    dist = abs(ep['pos'][0] - next_pos[0]) + abs(ep['pos'][1] - next_pos[1])
                    weight = np.exp(-dist * 0.2)
                    
                    # Combined score
                    ep_score = (goal_signal * 0.6 + success_signal * 0.4) * weight
                    score += ep_score
            
            # Normalize by number of episodes
            if episode_ids:
                score /= len(episode_ids)
            
            # Small bonus for larger hop count (exploration)
            score += n_hops * 0.05
            
            if score > best_score:
                best_score = score
                best_hop = n_hops
        
        self.hop_selections[f'{best_hop}-hop'] += 1
        return best_score, best_hop
    
    def decide_action(self) -> str:
        """Decide action using multi-hop evaluation"""
        visual = self.visual_memory.get(self.position, {})
        
        action_scores = {}
        
        for action in ['up', 'right', 'down', 'left']:
            # Skip walls
            if visual.get(action) == 'wall':
                action_scores[action] = -10.0
                continue
            
            # Multi-hop evaluation
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
        """Navigate maze using pure episodic memory"""
        print(f"\nPure Episodic Navigation (No Visit Counts)")
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
            
            # Execute action
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
        print(f"Path efficiency: {len(set(self.path))/steps*100:.1f}%")
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
            'path_efficiency': len(set(self.path))/steps*100,
            'time': elapsed,
            'hop_selections': dict(self.hop_selections)
        }


def create_complex_maze(size: int, seed: int = 42) -> np.ndarray:
    """Create a complex maze with loops"""
    np.random.seed(seed)
    maze = np.ones((size, size), dtype=int)
    
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
    
    # Start carving
    carve(1, 1)
    
    # Ensure goal is reachable
    maze[size-2, size-2] = 0
    
    # Add some loops for complexity
    loop_count = int(size * 0.8)
    for _ in range(loop_count):
        x = np.random.randint(2, size-2)
        y = np.random.randint(2, size-2)
        if maze[y, x] == 1:
            # Check if creating a path here would create a loop
            neighbors = sum(1 for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]
                          if 0 <= x+dx < size and 0 <= y+dy < size and maze[y+dy, x+dx] == 0)
            if neighbors >= 2:
                maze[y, x] = 0
    
    return maze


def visualize_maze_with_path(maze: np.ndarray, path: List[Tuple[int, int]], 
                           filename: str = None):
    """Visualize maze with the solution path"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw maze
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            if maze[y, x] == 1:
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor='black'))
    
    # Draw path
    if path:
        path_x = [p[0] + 0.5 for p in path]
        path_y = [p[1] + 0.5 for p in path]
        ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.5)
        
        # Mark start and end
        ax.plot(path_x[0], path_y[0], 'go', markersize=10, label='Start')
        ax.plot(path_x[-1], path_y[-1], 'ro', markersize=10, label='Goal')
    
    ax.set_xlim(0, maze.shape[1])
    ax.set_ylim(0, maze.shape[0])
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend()
    
    plt.title(f"Maze Solution ({len(path)} steps)")
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {filename}")
    
    plt.close()


def test_pure_episodic():
    """Test pure episodic navigation"""
    print("="*70)
    print("PURE EPISODIC NAVIGATION TEST")
    print("Without visit counts - using only episodic memory")
    print("="*70)
    
    # Test on different maze sizes
    for size in [25, 30]:
        print(f"\n{'='*50}")
        print(f"Testing {size}×{size} maze")
        print('='*50)
        
        # Create maze
        maze = create_complex_maze(size)
        
        # Create navigator
        nav = PureEpisodicNavigator(maze, message_depth=3)
        
        # Navigate
        result = nav.navigate(max_steps=3000)
        
        # Visualize if successful
        if result['success']:
            os.makedirs('visualizations', exist_ok=True)
            visualize_maze_with_path(
                maze, nav.path, 
                f'visualizations/pure_episodic_{size}x{size}.png'
            )


if __name__ == "__main__":
    test_pure_episodic()