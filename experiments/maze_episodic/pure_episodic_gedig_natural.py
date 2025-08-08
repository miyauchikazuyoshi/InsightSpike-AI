#!/usr/bin/env python3
"""
Pure Episodic Navigator with Natural geDIG-based Multi-hop Selection
- No artificial selection based on stuck_counter or norm
- Let geDIG values naturally determine the optimal depth
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add path for integrated index
sys.path.append(os.path.join(os.path.dirname(__file__), '../maze-optimized-search/src'))
from gedig_aware_integrated_index import GeDIGAwareIntegratedIndex


class PureEpisodicGeDIGNatural:
    """Navigator with natural geDIG-based depth selection"""
    
    def __init__(self, maze: np.ndarray, max_depth: int = 10):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = self._find_start()
        self.goal = self._find_goal()
        self.max_depth = max_depth
        
        # geDIG-aware integrated index
        self.index = GeDIGAwareIntegratedIndex(
            dimension=6,
            config={
                'similarity_threshold': 0.4,  # Lower for more connections
                'gedig_threshold': 0.7,
                'gedig_weight': 0.3,  # Balanced weight
                'max_edges_per_node': 20  # More edges for deep propagation
            }
        )
        
        # Memory systems
        self.visual_memory = {}
        self.search_times = []
        self.path = [(self.position[0], self.position[1])]
        self.recent_positions = []
        
        # Action mapping
        self.actions = ['up', 'right', 'down', 'left']
        self.action_deltas = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        
        # Multi-hop statistics (1-10 hop)
        self.hop_contributions = {f'{i}-hop': [] for i in range(1, self.max_depth+1)}
        self.depth_weights = []
    
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
    
    def _update_visual_memory(self):
        x, y = self.position
        if (x, y) not in self.visual_memory:
            self.visual_memory[(x, y)] = {}
            
            for action, (dx, dy) in self.action_deltas.items():
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.height and 0 <= ny < self.width:
                    self.visual_memory[(x, y)][action] = 'path' if self.maze[nx, ny] == 0 else 'wall'
    
    def _create_episode_vector(self, x: int, y: int) -> np.ndarray:
        """Create episode vector with visual information"""
        vec = np.zeros(6, dtype=np.float32)
        
        # Normalized position
        vec[0] = x / self.height
        vec[1] = y / self.width
        
        # Visual features
        vis = self.visual_memory.get((x, y), {})
        for i, action in enumerate(self.actions):
            if vis.get(action) == 'path':
                vec[2+i] = 1.0
            elif vis.get(action) == 'wall':
                vec[2+i] = -1.0
        
        return vec
    
    def _create_query_vector(self, x: int, y: int) -> np.ndarray:
        """Create query vector seeking paths and goal"""
        vec = np.zeros(6, dtype=np.float32)
        
        # Current position
        vec[0] = x / self.height
        vec[1] = y / self.width
        
        # Seek paths in all directions
        for i in range(4):
            vec[2+i] = 1.0
        
        # Add subtle goal bias (not too strong to maintain naturalness)
        goal_dx = self.goal[0] - x
        goal_dy = self.goal[1] - y
        goal_dist = abs(goal_dx) + abs(goal_dy)
        
        # Very subtle directional preference
        if goal_dist > 0:
            if abs(goal_dx) > abs(goal_dy):
                if goal_dx < 0:
                    vec[2] *= 1.2  # Slight up preference
                else:
                    vec[3] *= 1.2  # Slight down preference
            else:
                if goal_dy < 0:
                    vec[4] *= 1.2  # Slight left preference
                else:
                    vec[5] *= 1.2  # Slight right preference
        
        return vec
    
    def _message_passing_with_gedig_quality(self, indices: List[int], depth: int) -> Tuple[np.ndarray, float]:
        """
        Message passing that returns both direction and quality score
        Quality score reflects how good the geDIG paths are
        """
        if depth <= 0 or not indices:
            return np.zeros(6), 0.0
        
        # Initialize messages
        messages = {}
        path_qualities = {}  # Track quality of paths
        
        for i, idx in enumerate(indices):
            messages[idx] = 1.0 / (i + 1)
            path_qualities[idx] = 1.0
        
        # Track all visited nodes
        all_visited = set(indices)
        
        # Propagate through graph
        for d in range(depth):
            new_messages = {}
            new_qualities = {}
            decay_factor = 0.8 ** d
            
            for node, value in messages.items():
                if node not in self.index.graph:
                    continue
                
                # Self-loop with decay
                if d < depth - 1:
                    new_messages[node] = value * 0.7 * decay_factor
                    new_qualities[node] = path_qualities.get(node, 0.5) * 0.9
                
                # Propagate to neighbors
                for neighbor in self.index.graph.neighbors(node):
                    if neighbor in all_visited and d < depth - 1:
                        continue
                    
                    edge_data = self.index.graph[node][neighbor]
                    
                    # Get edge quality (combines similarity and geDIG)
                    weight = edge_data.get('weight', 0.5)
                    gedig = edge_data.get('gedig', 1.0)
                    
                    # Edge quality: high weight, low geDIG is best
                    edge_quality = weight * (1.0 - gedig)
                    
                    # Propagate value and quality
                    propagation = value * edge_quality * decay_factor
                    path_quality = path_qualities.get(node, 0.5) * edge_quality
                    
                    if neighbor in new_messages:
                        if propagation > new_messages[neighbor]:
                            new_messages[neighbor] = propagation
                            new_qualities[neighbor] = path_quality
                    else:
                        new_messages[neighbor] = propagation
                        new_qualities[neighbor] = path_quality
                    
                    all_visited.add(neighbor)
            
            messages = new_messages
            path_qualities = new_qualities
            
            if not messages:
                break
        
        # Aggregate into direction vector
        direction = np.zeros(6)
        total_weight = 0
        total_quality = 0
        
        for idx, value in messages.items():
            if idx < len(self.index.metadata):
                episode = self.index.metadata[idx]
                vec = episode['vec']
                
                # Consider path quality in weighting
                quality = path_qualities.get(idx, 0.5)
                
                # Weight by success
                if episode.get('success', False):
                    success_weight = 2.0
                elif episode.get('hit_wall', False):
                    success_weight = 0.3
                else:
                    success_weight = 1.0
                
                weight = value * success_weight * quality
                direction += vec * weight
                total_weight += weight
                total_quality += quality * value
        
        # Normalize direction
        if total_weight > 0:
            direction = direction / total_weight
        
        # Calculate overall quality score for this depth
        quality_score = total_quality / max(len(messages), 1)
        
        return direction, quality_score
    
    def get_action(self) -> Optional[str]:
        """Get action using natural geDIG-based depth selection"""
        self._update_visual_memory()
        
        x, y = self.position
        current_vec = self._create_episode_vector(x, y)
        query_vec = self._create_query_vector(x, y)
        
        # Track position
        self.recent_positions.append((x, y))
        if len(self.recent_positions) > 50:
            self.recent_positions.pop(0)
        
        # Store current episode
        episode_idx = self.index.add_episode({
            'vec': current_vec,
            'text': f"Episode at ({x}, {y})",
            'pos': (x, y),
            'c_value': 0.7,
            'distance_to_goal': abs(x - self.goal[0]) + abs(y - self.goal[1])
        })
        
        # Search with increasing k for different depths
        start_time = time.time()
        
        # Calculate all depths (1 to max_depth)
        directions_and_qualities = []
        
        for depth in range(1, self.max_depth + 1):
            # Increase k with depth
            k = min(20 + depth * 10, 100)
            
            # Search
            indices, scores = self.index.search(query_vec, k=k, mode='hybrid')
            
            # Message passing with quality tracking
            direction, quality = self._message_passing_with_gedig_quality(indices.tolist(), depth)
            directions_and_qualities.append((direction, quality, depth))
        
        search_time = (time.time() - start_time) * 1000
        self.search_times.append(search_time)
        
        # Natural selection based on geDIG quality
        # Higher quality means better geDIG paths were found
        qualities = [q for _, q, _ in directions_and_qualities]
        
        # Normalize qualities
        total_quality = sum(qualities) + 0.001
        weights = [q / total_quality for q in qualities]
        
        # Record weights for analysis
        self.depth_weights.append(weights)
        
        # Combine directions weighted by their geDIG quality
        final_direction = np.zeros(6)
        for i, (direction, quality, depth) in enumerate(directions_and_qualities):
            final_direction += direction * weights[i]
            
            # Track contribution
            contribution = np.linalg.norm(direction * weights[i])
            self.hop_contributions[f'{depth}-hop'].append(contribution)
        
        # Normalize final direction
        norm = np.linalg.norm(final_direction)
        if norm > 0:
            final_direction = final_direction / norm
        
        # Convert to action scores
        action_scores = np.zeros(4)
        
        for i, action in enumerate(self.actions):
            dx, dy = self.action_deltas[action]
            
            # Position-based score
            action_vec = np.zeros(2)
            action_vec[0] = dx / self.height
            action_vec[1] = dy / self.width
            position_score = np.dot(final_direction[:2], action_vec)
            
            # Feature-based score (visual paths)
            feature_score = final_direction[2+i] if final_direction[2+i] > 0 else 0
            
            # Natural combination
            action_scores[i] = max(0, position_score * 0.6 + feature_score * 0.4)
        
        # Minimal noise (just to break ties)
        action_scores += np.random.random(4) * 0.001
        
        # Select action
        if np.sum(action_scores) > 0:
            probs = action_scores / np.sum(action_scores)
            return np.random.choice(self.actions, p=probs)
        
        return np.random.choice(self.actions)
    
    def move(self, action: str) -> bool:
        """Execute action and store result"""
        if action not in self.actions:
            return False
            
        dx, dy = self.action_deltas[action]
        new_x, new_y = self.position[0] + dx, self.position[1] + dy
        
        # Try to move
        if (0 <= new_x < self.height and 0 <= new_y < self.width and 
            self.maze[new_x, new_y] == 0):
            
            # Store successful move
            if len(self.index.metadata) > 0:
                self.index.metadata[-1]['action'] = action
                self.index.metadata[-1]['success'] = True
            
            self.position = (new_x, new_y)
            self.path.append(self.position)
            return True
        else:
            # Store failed move
            if len(self.index.metadata) > 0:
                self.index.metadata[-1]['action'] = action
                self.index.metadata[-1]['success'] = False
                self.index.metadata[-1]['hit_wall'] = True
            return False
    
    def navigate(self, max_steps: int = 5000) -> Dict:
        """Navigate maze using natural geDIG selection"""
        start_time = time.time()
        wall_hits = 0
        
        for step in range(max_steps):
            if self.position == self.goal:
                total_time = time.time() - start_time
                stats = self.index.get_statistics()
                
                print(f"\n🎉 SUCCESS! Natural geDIG selection worked!")
                
                return {
                    'success': True,
                    'steps': step,
                    'total_episodes': len(self.index.metadata),
                    'total_time': total_time,
                    'avg_search_time': np.mean(self.search_times) if self.search_times else 0,
                    'search_times': self.search_times,
                    'path': self.path,
                    'hop_contributions': self.hop_contributions,
                    'depth_weights': self.depth_weights,
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
                hit_rate = wall_hits/step*100
                dist = abs(self.position[0]-self.goal[0])+abs(self.position[1]-self.goal[1])
                
                print(f"Step {step}: pos={self.position}, dist={dist}, "
                      f"episodes={stats['episodes']}, edges={stats['edges']}, "
                      f"wall_hits={wall_hits} ({hit_rate:.1f}%)")
                
                # Show dominant depths
                if step % 500 == 0:
                    avg_weights = np.mean(self.depth_weights[-100:], axis=0)
                    dominant_depths = np.argsort(avg_weights)[-3:][::-1]
                    print(f"  Dominant depths: {[d+1 for d in dominant_depths]} "
                          f"with weights {[avg_weights[d] for d in dominant_depths]}")
        
        total_time = time.time() - start_time
        stats = self.index.get_statistics()
        
        return {
            'success': False,
            'steps': max_steps,
            'total_episodes': len(self.index.metadata),
            'total_time': total_time,
            'avg_search_time': np.mean(self.search_times) if self.search_times else 0,
            'search_times': self.search_times,
            'path': self.path,
            'hop_contributions': self.hop_contributions,
            'depth_weights': self.depth_weights,
            'index_stats': stats,
            'wall_hits': wall_hits
        }