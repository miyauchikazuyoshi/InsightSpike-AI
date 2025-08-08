#!/usr/bin/env python3
"""
Pure Episodic Navigator with Integrated Index
=============================================

Reimplementation of PureEpisodicNavigator using IntegratedVectorGraphIndex
to solve the O(n²) bottleneck problem.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import sys

# Add the path to access integrated index implementation
sys.path.append(os.path.join(os.path.dirname(__file__), '../maze-optimized-search/src'))
from integrated_vector_graph_index import IntegratedVectorGraphIndex

class PureEpisodicIntegrated:
    """Pure episodic memory navigator with integrated index for O(1) search"""
    
    def __init__(self, maze: np.ndarray, message_depth: int = 3):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.goal = (self.width-2, self.height-2)
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        
        # Integrated index for efficient episode storage and search
        self.index = IntegratedVectorGraphIndex(
            dimension=6,  # 4 actions + 2 position
            config={
                'similarity_threshold': 0.8,
                'use_faiss': False,  # Start with numpy for small mazes
                'faiss_threshold': 5000
            }
        )
        
        self.message_depth = message_depth
        
        # Visual memory (local observation)
        self.visual_memory = {}
        self._update_visual_memory(1, 1)
        
        # Statistics
        self.wall_hits = 0
        self.moves = 0
        self.hop_selections = {'1-hop': 0, '2-hop': 0, '3-hop': 0}
        self.search_times = []
        
    def _update_visual_memory(self, x: int, y: int):
        """Update visual memory from current position"""
        self.visual_memory[(x, y)] = {}
        for action, (dx, dy) in {'up': (0, -1), 'right': (1, 0), 
                                'down': (0, 1), 'left': (-1, 0)}.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                self.visual_memory[(x, y)][action] = 'path' if self.maze[ny, nx] == 0 else 'wall'
    
    def _create_episode_vector(self, x: int, y: int) -> np.ndarray:
        """Create episode vector from position"""
        vec = np.zeros(6)
        
        # Position features (normalized)
        vec[0] = x / self.width
        vec[1] = y / self.height
        
        # Visual features (walls/paths in 4 directions)
        vis = self.visual_memory.get((x, y), {})
        for i, action in enumerate(['up', 'right', 'down', 'left']):
            if vis.get(action) == 'path':
                vec[2+i] = 1.0
            elif vis.get(action) == 'wall':
                vec[2+i] = -1.0
        
        return vec
    
    def _message_passing(self, source_indices: List[int], depth: int) -> np.ndarray:
        """Deep message passing from source episodes using integrated index"""
        if depth <= 0 or not source_indices:
            return np.zeros(6)
        
        # Get graph from integrated index
        graph = self.index.graph
        
        # Initialize messages
        messages = {}
        for idx in source_indices:
            messages[idx] = 1.0
        
        # Propagate messages through graph
        for _ in range(depth):
            new_messages = {}
            for node, value in messages.items():
                # Propagate to neighbors
                if node in graph:
                    neighbors = list(graph.neighbors(node))
                    for neighbor in neighbors:
                        edge_data = graph[node][neighbor]
                        weight = edge_data.get('weight', 0.5)
                        if neighbor not in new_messages:
                            new_messages[neighbor] = 0
                        new_messages[neighbor] += value * weight * 0.7
            
            # Merge with existing messages
            for node, value in new_messages.items():
                if node not in messages:
                    messages[node] = 0
                messages[node] = max(messages[node], value)
        
        # Aggregate messages into direction vector
        direction = np.zeros(6)
        for idx, value in messages.items():
            if value > 0.1:  # Threshold
                episode = self.index.get_episode(idx)
                vec = episode['vec']
                direction += vec * value
        
        return direction / (np.linalg.norm(direction) + 1e-8)
    
    def get_action(self) -> str:
        """Get next action using efficient multi-hop evaluation"""
        if self.position == self.goal:
            return None
            
        x, y = self.position
        current_vec = self._create_episode_vector(x, y)
        
        # Store current episode in integrated index
        episode_idx = self.index.add_episode({
            'vec': current_vec,
            'text': f"Episode at ({x}, {y})",
            'pos': (x, y),
            'c_value': 0.5  # Base confidence
        })
        
        # Efficient search using integrated index (O(1) instead of O(n²))
        start_time = time.time()
        
        # 1-hop: Direct similar episodes
        indices_1hop, scores_1hop = self.index.search(current_vec, k=10, mode='vector')
        
        # 2-hop: Extended through graph
        indices_2hop, scores_2hop = self.index.search(current_vec, k=20, mode='hybrid')
        
        # 3-hop: Spatial + vector search for exploration
        indices_3hop, scores_3hop = self.index.search(
            current_vec, k=30, mode='hybrid',
            spatial_center=(x, y), spatial_radius=5
        )
        
        search_time = time.time() - start_time
        self.search_times.append(search_time * 1000)  # Convert to ms
        
        # Goal signal detection
        goal_vec = self._create_episode_vector(*self.goal)
        goal_indices, goal_scores = self.index.search(goal_vec, k=5)
        
        # Multi-hop evaluation
        direction_1hop = self._message_passing(indices_1hop.tolist(), 1)
        direction_2hop = self._message_passing(indices_2hop.tolist(), 2)
        direction_3hop = self._message_passing(indices_3hop.tolist(), 3)
        
        # Goal pull
        goal_direction = np.zeros(6)
        if len(goal_indices) > 0 and goal_scores[0] > 0.7:
            goal_direction = self._message_passing(goal_indices.tolist(), self.message_depth)
        
        # Adaptive selection based on exploration needs
        if np.linalg.norm(direction_1hop) > 0.5:
            direction = direction_1hop
            self.hop_selections['1-hop'] += 1
        elif np.linalg.norm(direction_2hop) > 0.3:
            direction = direction_2hop
            self.hop_selections['2-hop'] += 1
        else:
            direction = direction_3hop
            self.hop_selections['3-hop'] += 1
        
        # Combine with goal pull
        direction = 0.7 * direction + 0.3 * goal_direction
        
        # Convert to action
        return self._vector_to_action(direction, x, y)
    
    def _vector_to_action(self, direction: np.ndarray, x: int, y: int) -> str:
        """Convert direction vector to concrete action"""
        # Extract directional components
        action_scores = {
            'up': direction[2] if direction[2] > 0 else 0,
            'right': direction[3] if direction[3] > 0 else 0,
            'down': direction[4] if direction[4] > 0 else 0,
            'left': direction[5] if direction[5] > 0 else 0
        }
        
        # Add goal bias
        gx, gy = self.goal
        if gx > x:
            action_scores['right'] += 0.3
        elif gx < x:
            action_scores['left'] += 0.3
        if gy > y:
            action_scores['down'] += 0.3
        elif gy < y:
            action_scores['up'] += 0.3
        
        # Filter out walls
        vis = self.visual_memory.get((x, y), {})
        valid_actions = []
        for action, score in action_scores.items():
            if vis.get(action) == 'path':
                valid_actions.append((action, score))
        
        if not valid_actions:
            # Exploration when stuck
            possible = []
            for action in ['up', 'right', 'down', 'left']:
                if vis.get(action) == 'path':
                    possible.append(action)
            return np.random.choice(possible) if possible else 'up'
        
        # Choose best valid action
        valid_actions.sort(key=lambda x: x[1], reverse=True)
        return valid_actions[0][0]
    
    def move(self, action: str) -> bool:
        """Execute action and update position"""
        x, y = self.position
        dx, dy = {'up': (0, -1), 'right': (1, 0), 
                  'down': (0, 1), 'left': (-1, 0)}[action]
        
        nx, ny = x + dx, y + dy
        
        if self.maze[ny, nx] == 1:  # Wall
            self.wall_hits += 1
            return False
        
        self.position = (nx, ny)
        self.visited.add((nx, ny))
        self.path.append((nx, ny))
        self.moves += 1
        
        # Update visual memory at new position
        self._update_visual_memory(nx, ny)
        
        return True
    
    def navigate(self, max_steps: int = 10000) -> Dict:
        """Navigate maze and return results"""
        start_time = time.time()
        
        for step in range(max_steps):
            if self.position == self.goal:
                success = True
                break
                
            action = self.get_action()
            if action is None:
                success = True
                break
                
            self.move(action)
            
            if step % 100 == 0:
                episodes = len(self.index.metadata)
                avg_search = np.mean(self.search_times[-100:]) if self.search_times else 0
                print(f"Step {step}: pos={self.position}, episodes={episodes}, "
                      f"avg_search={avg_search:.2f}ms")
        else:
            success = False
        
        total_time = time.time() - start_time
        
        return {
            'success': success,
            'steps': self.moves,
            'visited_count': len(self.visited),
            'wall_hits': self.wall_hits,
            'total_time': total_time,
            'path': self.path,
            'hop_selections': self.hop_selections,
            'total_episodes': len(self.index.metadata),
            'avg_search_time': np.mean(self.search_times) if self.search_times else 0,
            'search_times': self.search_times
        }
    
    def visualize(self, save_path: str = None):
        """Visualize the maze and path"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Maze and path
        ax1.imshow(self.maze, cmap='binary')
        
        # Draw path
        if self.path:
            path_x = [p[0] for p in self.path]
            path_y = [p[1] for p in self.path]
            ax1.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7)
            ax1.plot(path_x[0], path_y[0], 'go', markersize=10, label='Start')
            ax1.plot(path_x[-1], path_y[-1], 'ro', markersize=10, label='End')
        
        # Mark goal
        ax1.plot(self.goal[0], self.goal[1], 'r*', markersize=15, label='Goal')
        
        ax1.set_title(f'Maze Navigation (Integrated Index)\nSteps: {self.moves}, Episodes: {len(self.index.metadata)}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Search time analysis
        if self.search_times:
            ax2.plot(self.search_times)
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Search Time (ms)')
            ax2.set_title(f'Search Performance\nAvg: {np.mean(self.search_times):.2f}ms')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()


def test_integrated_navigator():
    """Test the integrated navigator on various maze sizes"""
    # Use the maze generator from maze-optimized-search
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
    from insightspike.environments.proper_maze_generator import ProperMazeGenerator
    
    generator = ProperMazeGenerator()
    
    # Test on different sizes
    for size in [(15, 15)]:
        print(f"\n{'='*60}")
        print(f"Testing {size[0]}x{size[1]} maze with Integrated Index")
        print(f"{'='*60}")
        
        maze = generator.generate_dfs_maze(size=size)
        navigator = PureEpisodicIntegrated(maze)
        
        result = navigator.navigate()
        
        print(f"\nResults:")
        print(f"  Success: {result['success']}")
        print(f"  Steps: {result['steps']}")
        print(f"  Episodes: {result['total_episodes']}")
        print(f"  Avg Search Time: {result['avg_search_time']:.2f}ms")
        print(f"  Total Time: {result['total_time']:.2f}s")
        print(f"  Hop usage: {result['hop_selections']}")
        
        # Compare with O(n²) estimation
        n = result['total_episodes']
        estimated_old_time = (n * n) / 1000  # Rough O(n²) estimate in ms
        speedup = estimated_old_time / result['avg_search_time']
        print(f"  Estimated speedup: {speedup:.1f}x vs O(n²)")
        
        # Save visualization
        save_path = f"pure_episodic_integrated_{size[0]}x{size[1]}.png"
        navigator.visualize(save_path)
        
        if not result['success'] and size[0] <= 25:
            print("  WARNING: Failed on a size that should succeed!")


if __name__ == "__main__":
    test_integrated_navigator()