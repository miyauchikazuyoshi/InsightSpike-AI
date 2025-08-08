#!/usr/bin/env python3
"""
Pure Episodic Navigator with Movement Episode Memory
- Stores movement episodes: (x, y, direction, success, wall/path, visit_count, goal)
- Visual info added as pre-episodes for 4 directions
- Query searches for successful movements from current position
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add path for integrated index
sys.path.append(os.path.join(os.path.dirname(__file__), '../maze-optimized-search/src'))
from gedig_aware_integrated_index import GeDIGAwareIntegratedIndex


class PureEpisodicMovementMemory:
    """Navigator that stores movement episodes"""
    
    def __init__(self, maze: np.ndarray, max_depth: int = 7):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = self._find_start()
        self.goal = self._find_goal()
        self.max_depth = max_depth
        
        # geDIG-aware integrated index with 7 dimensions
        # (x, y, direction, success, wall/path, visit_count, goal)
        self.index = GeDIGAwareIntegratedIndex(
            dimension=7,
            config={
                'similarity_threshold': 0.5,
                'gedig_threshold': 0.6,
                'gedig_weight': 0.35,
                'max_edges_per_node': 15
            }
        )
        
        # Memory systems
        self.visit_counts = {}  # Track visit counts per position
        self.search_times = []
        self.path = [(self.position[0], self.position[1])]
        self.recent_positions = []
        
        # Action mapping
        self.actions = ['up', 'right', 'down', 'left']
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        self.action_deltas = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        
        # Multi-hop statistics
        self.hop_contributions = {f'{i}-hop': [] for i in range(1, self.max_depth+1)}
        self.depth_weights = []
        
        # Episode counter for tracking
        self.episode_count = 0
    
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
    
    def _update_visit_count(self, pos: Tuple[int, int] = None):
        """Update visit count for position"""
        if pos is None:
            pos = self.position
        if pos not in self.visit_counts:
            self.visit_counts[pos] = 0
        self.visit_counts[pos] += 1
    
    def _create_movement_episode(self, x: int, y: int, direction: str, 
                                 success: bool, is_wall: bool) -> np.ndarray:
        """Create movement episode vector"""
        vec = np.zeros(7, dtype=np.float32)
        
        # Position (normalized)
        vec[0] = x / self.height
        vec[1] = y / self.width
        
        # Direction (one-hot encoded as single value)
        vec[2] = self.action_to_idx[direction] / 3.0  # Normalize to [0, 1]
        
        # Success (1 for success, 0 for failure)
        vec[3] = 1.0 if success else 0.0
        
        # Wall/Path (-1 for wall, 1 for path, 0 for unknown)
        vec[4] = -1.0 if is_wall else (1.0 if not is_wall else 0.0)
        
        # Visit count (log normalized)
        visit_count = self.visit_counts.get((x, y), 0)
        vec[5] = np.log1p(visit_count) / 10.0
        
        # Goal (1 if at goal, 0 otherwise)
        vec[6] = 1.0 if (x, y) == self.goal else 0.0
        
        return vec
    
    def _create_visual_episode(self, x: int, y: int, direction: str) -> np.ndarray:
        """Create visual observation episode for a direction"""
        vec = np.zeros(7, dtype=np.float32)
        
        # Position
        vec[0] = x / self.height
        vec[1] = y / self.width
        
        # Direction
        vec[2] = self.action_to_idx[direction] / 3.0
        
        # Success is unknown for visual episodes
        vec[3] = 0.5  # Neutral value
        
        # Check if wall or path
        dx, dy = self.action_deltas[direction]
        nx, ny = x + dx, y + dy
        
        if 0 <= nx < self.height and 0 <= ny < self.width:
            vec[4] = 1.0 if self.maze[nx, ny] == 0 else -1.0
        else:
            vec[4] = -1.0  # Out of bounds = wall
        
        # Visit count
        visit_count = self.visit_counts.get((x, y), 0)
        vec[5] = np.log1p(visit_count) / 10.0
        
        # Goal check for the observed position
        vec[6] = 1.0 if (nx, ny) == self.goal else 0.0
        
        return vec
    
    def _add_visual_observations(self):
        """Add visual observation episodes for all 4 directions"""
        x, y = self.position
        
        for direction in self.actions:
            vec = self._create_visual_episode(x, y, direction)
            
            # Add as episode with high geDIG value for 1-hop connection
            self.index.add_episode({
                'vec': vec,
                'text': f"Visual: from ({x},{y}) looking {direction}",
                'pos': (x, y),
                'type': 'visual',
                'direction': direction,
                'c_value': 0.9,  # High quality for visual info
                'episode_id': self.episode_count
            })
            self.episode_count += 1
    
    def _create_query_vector(self) -> np.ndarray:
        """Create query for finding successful movements from current position"""
        vec = np.zeros(7, dtype=np.float32)
        
        x, y = self.position
        
        # Current position
        vec[0] = x / self.height
        vec[1] = y / self.width
        
        # Direction: null (don't care which direction, represented as 0.5)
        vec[2] = 0.5
        
        # Success: want successful moves
        vec[3] = 1.0
        
        # Wall/Path: null (will be filled by visual observations)
        vec[4] = 0.0
        
        # Visit count: just current count, no preference
        current_visits = self.visit_counts.get((x, y), 0)
        vec[5] = np.log1p(current_visits) / 10.0  # Just encode current state
        
        # Goal: neutral (let memory decide)
        vec[6] = 0.5
        
        return vec
    
    def _message_passing_with_quality(self, indices: List[int], depth: int) -> Tuple[np.ndarray, float]:
        """Message passing with geDIG quality tracking"""
        if depth <= 0 or not indices:
            return np.zeros(7), 0.0
        
        # Initialize messages
        messages = {}
        path_qualities = {}
        
        for i, idx in enumerate(indices):
            messages[idx] = 1.0 / (i + 1)
            path_qualities[idx] = 1.0
        
        # Propagate through graph
        for d in range(depth):
            new_messages = {}
            new_qualities = {}
            decay_factor = 0.85 ** d
            
            for node, value in messages.items():
                if node not in self.index.graph:
                    continue
                
                # Self-loop
                if d < depth - 1:
                    new_messages[node] = value * 0.8 * decay_factor
                    new_qualities[node] = path_qualities.get(node, 0.5) * 0.95
                
                # Propagate to neighbors
                for neighbor in self.index.graph.neighbors(node):
                    edge_data = self.index.graph[node][neighbor]
                    
                    # Edge quality
                    weight = edge_data.get('weight', 0.5)
                    gedig = edge_data.get('gedig', 1.0)
                    edge_quality = weight * (1.0 - gedig * 0.5)
                    
                    # Propagate
                    propagation = value * edge_quality * decay_factor
                    path_quality = path_qualities.get(node, 0.5) * edge_quality
                    
                    if neighbor in new_messages:
                        if propagation > new_messages[neighbor]:
                            new_messages[neighbor] = propagation
                            new_qualities[neighbor] = path_quality
                    else:
                        new_messages[neighbor] = propagation
                        new_qualities[neighbor] = path_quality
            
            messages = new_messages
            path_qualities = new_qualities
            
            if not messages:
                break
        
        # Aggregate movement episodes
        direction_scores = np.zeros(4)
        total_weight = 0
        total_quality = 0
        
        for idx, value in messages.items():
            if idx < len(self.index.metadata):
                episode = self.index.metadata[idx]
                vec = episode['vec']
                
                # Only consider movement episodes (not visual)
                if episode.get('type') != 'visual':
                    # Extract direction from episode
                    direction_val = vec[2] * 3.0  # Denormalize
                    direction_idx = int(round(direction_val))
                    
                    if 0 <= direction_idx < 4:
                        # Pure memory-based: only use success and message value
                        success = vec[3]
                        quality = path_qualities.get(idx, 0.5)
                        
                        # No bonuses or penalties - pure memory
                        weight = value * success * quality
                        direction_scores[direction_idx] += weight
                        total_weight += weight
                        total_quality += quality * value
        
        # Normalize to create final direction vector
        direction_vec = np.zeros(7)
        if total_weight > 0:
            # Convert direction scores to full vector
            best_direction = np.argmax(direction_scores)
            direction_vec[2] = best_direction / 3.0
            direction_vec[3] = 1.0  # Expect success
        
        quality_score = total_quality / max(len(messages), 1)
        
        return direction_vec, quality_score
    
    def _get_direction_scores_from_episodes(self, indices: List[int], scores: np.ndarray) -> np.ndarray:
        """Get direction scores from retrieved episodes"""
        direction_scores = np.zeros(4)
        
        for i, idx in enumerate(indices):
            if idx < len(self.index.metadata):
                episode = self.index.metadata[idx]
                
                # Only count successful movement episodes
                if episode.get('type') == 'movement' and episode.get('success'):
                    action = episode.get('action')
                    if action in self.actions:
                        action_idx = self.action_to_idx[action]
                        # Weight by retrieval score
                        direction_scores[action_idx] += scores[i] if i < len(scores) else 1.0
        
        return direction_scores
    
    def get_action(self) -> Optional[str]:
        """Get action using movement episode memory"""
        self._update_visit_count()
        
        # Add visual observations for current position
        self._add_visual_observations()
        
        # Create query
        query_vec = self._create_query_vector()
        
        x, y = self.position
        
        # Track position
        self.recent_positions.append((x, y))
        if len(self.recent_positions) > 50:
            self.recent_positions.pop(0)
        
        # Search and aggregate
        start_time = time.time()
        
        search_time = (time.time() - start_time) * 1000
        self.search_times.append(search_time)
        
        # Collect all direction scores from each depth
        all_direction_scores = []
        
        for depth in range(1, self.max_depth + 1):
            k = min(20 + depth * 5, 80)
            indices, scores = self.index.search(query_vec, k=k, mode='hybrid')
            
            # Get direction scores for this depth
            direction_scores = self._get_direction_scores_from_episodes(indices.tolist(), scores)
            all_direction_scores.append(direction_scores)
            
            # Track contributions
            total_score = np.sum(direction_scores)
            self.hop_contributions[f'{depth}-hop'].append(total_score)
        
        # Aggregate scores from all depths
        final_scores = np.zeros(4)
        for i, scores in enumerate(all_direction_scores):
            weight = 1.0 / (i + 1)  # Decay by depth
            final_scores += scores * weight
        
        # Pure memory-based selection
        if np.sum(final_scores) > 0:
            # Normalize to probabilities
            action_probs = final_scores / np.sum(final_scores)
        else:
            # If no memory, uniform random
            action_probs = np.ones(4) / 4
        
        return np.random.choice(self.actions, p=action_probs)
    
    def move(self, action: str) -> bool:
        """Execute action and store movement episode"""
        if action not in self.actions:
            return False
        
        x, y = self.position
        dx, dy = self.action_deltas[action]
        new_x, new_y = x + dx, y + dy
        
        # Try to move
        success = False
        is_wall = True
        
        if (0 <= new_x < self.height and 0 <= new_y < self.width and 
            self.maze[new_x, new_y] == 0):
            success = True
            is_wall = False
            self.position = (new_x, new_y)
            self.path.append(self.position)
        
        # Store movement episode
        movement_vec = self._create_movement_episode(x, y, action, success, is_wall)
        
        self.index.add_episode({
            'vec': movement_vec,
            'text': f"Move: from ({x},{y}) {action} -> {'success' if success else 'failed'}",
            'pos': (x, y),
            'type': 'movement',
            'action': action,
            'success': success,
            'c_value': 0.8 if success else 0.3,
            'episode_id': self.episode_count
        })
        self.episode_count += 1
        
        return success
    
    def navigate(self, max_steps: int = 5000) -> Dict:
        """Navigate using movement episode memory"""
        start_time = time.time()
        wall_hits = 0
        
        for step in range(max_steps):
            if self.position == self.goal:
                total_time = time.time() - start_time
                stats = self.index.get_statistics()
                
                print(f"\n🎉 SUCCESS with movement episode memory!")
                
                return {
                    'success': True,
                    'steps': step,
                    'total_episodes': self.episode_count,
                    'total_time': total_time,
                    'avg_search_time': np.mean(self.search_times) if self.search_times else 0,
                    'search_times': self.search_times,
                    'path': self.path,
                    'hop_contributions': self.hop_contributions,
                    'depth_weights': self.depth_weights,
                    'visit_counts': self.visit_counts,
                    'index_stats': stats,
                    'wall_hits': wall_hits
                }
            
            action = self.get_action()
            if action:
                success = self.move(action)
                if not success:
                    wall_hits += 1
            
            if step % 100 == 0 and step > 0:
                stats = self.index.get_statistics()
                hit_rate = wall_hits/step*100 if step > 0 else 0
                dist = abs(self.position[0]-self.goal[0])+abs(self.position[1]-self.goal[1])
                max_visits = max(self.visit_counts.values()) if self.visit_counts else 0
                
                print(f"Step {step}: pos={self.position}, dist={dist}, "
                      f"wall_hits={wall_hits} ({hit_rate:.1f}%), "
                      f"episodes={self.episode_count}, max_visits={max_visits}")
                
                if step % 500 == 0:
                    # Show episode type distribution
                    movement_eps = sum(1 for e in self.index.metadata 
                                     if e.get('type') == 'movement')
                    visual_eps = sum(1 for e in self.index.metadata 
                                   if e.get('type') == 'visual')
                    print(f"  Episodes: movement={movement_eps}, visual={visual_eps}")
        
        total_time = time.time() - start_time
        stats = self.index.get_statistics()
        
        return {
            'success': False,
            'steps': max_steps,
            'total_episodes': self.episode_count,
            'total_time': total_time,
            'avg_search_time': np.mean(self.search_times) if self.search_times else 0,
            'search_times': self.search_times,
            'path': self.path,
            'hop_contributions': self.hop_contributions,
            'depth_weights': self.depth_weights,
            'visit_counts': self.visit_counts,
            'index_stats': stats,
            'wall_hits': wall_hits
        }


def test_movement_memory():
    """Test movement episode memory navigator"""
    # Create simple maze
    maze = np.array([
        [0, 0, 1, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ])
    
    print("Testing Movement Episode Memory Navigator")
    print("Maze:")
    print(maze)
    print(f"Start: (0,0), Goal: (4,4)")
    
    navigator = PureEpisodicMovementMemory(maze)
    result = navigator.navigate(max_steps=500)
    
    if result['success']:
        print(f"\n✅ Solved in {result['steps']} steps")
        print(f"Total episodes: {result['total_episodes']}")
        print(f"Wall hits: {result['wall_hits']}")
        print(f"Path length: {len(result['path'])}")
    else:
        print(f"\n❌ Failed after {result['steps']} steps")
    
    return result


if __name__ == "__main__":
    test_movement_memory()