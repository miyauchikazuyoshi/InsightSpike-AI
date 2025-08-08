#!/usr/bin/env python3
"""
Pure Episodic Navigator with Visit Count Memory
- Episode vector includes visit count as additional dimension
- Natural exploration through visit memory
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add path for integrated index
sys.path.append(os.path.join(os.path.dirname(__file__), '../maze-optimized-search/src'))
from gedig_aware_integrated_index import GeDIGAwareIntegratedIndex


class PureEpisodicVisitCount:
    """Navigator with visit count in episodic memory"""
    
    def __init__(self, maze: np.ndarray, max_depth: int = 7):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = self._find_start()
        self.goal = self._find_goal()
        self.max_depth = max_depth
        
        # geDIG-aware integrated index with 7 dimensions (including visit count)
        self.index = GeDIGAwareIntegratedIndex(
            dimension=7,  # x, y, up, right, down, left, visit_count
            config={
                'similarity_threshold': 0.5,
                'gedig_threshold': 0.6,
                'gedig_weight': 0.35,
                'max_edges_per_node': 15
            }
        )
        
        # Memory systems
        self.visual_memory = {}
        self.visit_counts = {}  # Track visit counts per position
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
        
        # Multi-hop statistics
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
    
    def _update_visit_count(self):
        """Update visit count for current position"""
        pos = self.position
        if pos not in self.visit_counts:
            self.visit_counts[pos] = 0
        self.visit_counts[pos] += 1
    
    def _create_episode_vector(self, x: int, y: int) -> np.ndarray:
        """Create episode vector with visit count"""
        vec = np.zeros(7, dtype=np.float32)
        
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
        
        # Visit count (normalized by log scale to prevent domination)
        visit_count = self.visit_counts.get((x, y), 0)
        # Use log(1 + visits) to compress the range
        vec[6] = np.log1p(visit_count) / 10.0  # Normalized by expected max log visits
        
        return vec
    
    def _create_query_vector(self, x: int, y: int) -> np.ndarray:
        """Create query vector preferring less visited areas"""
        vec = np.zeros(7, dtype=np.float32)
        
        # Current position
        vec[0] = x / self.height
        vec[1] = y / self.width
        
        # Prefer paths in all directions
        for i in range(4):
            vec[2+i] = 1.0
        
        # Prefer less visited areas (negative value for visit count)
        # This creates natural exploration
        vec[6] = -0.5  # Want to find episodes with low visit counts
        
        # Subtle goal bias
        goal_dx = self.goal[0] - x
        goal_dy = self.goal[1] - y
        if abs(goal_dx) > abs(goal_dy):
            if goal_dx < 0:
                vec[2] *= 1.1
            else:
                vec[3] *= 1.1
        else:
            if goal_dy < 0:
                vec[4] *= 1.1
            else:
                vec[5] *= 1.1
        
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
        
        # Aggregate with visit count influence
        direction = np.zeros(7)
        total_weight = 0
        total_quality = 0
        
        for idx, value in messages.items():
            if idx < len(self.index.metadata):
                episode = self.index.metadata[idx]
                vec = episode['vec']
                
                # Quality and success weighting
                quality = path_qualities.get(idx, 0.5)
                success_weight = 2.0 if episode.get('success', False) else 0.5
                
                # Visit count influence - prefer less visited paths
                visit_dimension = vec[6] if len(vec) > 6 else 0
                exploration_bonus = np.exp(-visit_dimension * 2)  # Higher bonus for less visited
                
                weight = value * success_weight * quality * exploration_bonus
                direction += vec * weight
                total_weight += weight
                total_quality += quality * value
        
        # Normalize
        if total_weight > 0:
            direction = direction / total_weight
        
        quality_score = total_quality / max(len(messages), 1)
        
        return direction, quality_score
    
    def get_action(self) -> Optional[str]:
        """Get action using visit count memory"""
        self._update_visual_memory()
        self._update_visit_count()
        
        x, y = self.position
        current_vec = self._create_episode_vector(x, y)
        query_vec = self._create_query_vector(x, y)
        
        # Track position
        self.recent_positions.append((x, y))
        if len(self.recent_positions) > 50:
            self.recent_positions.pop(0)
        
        # Store episode with visit count
        episode_idx = self.index.add_episode({
            'vec': current_vec,
            'text': f"Episode at ({x}, {y}), visits: {self.visit_counts.get((x,y), 0)}",
            'pos': (x, y),
            'c_value': 0.7,
            'visit_count': self.visit_counts.get((x,y), 0),
            'distance_to_goal': abs(x - self.goal[0]) + abs(y - self.goal[1])
        })
        
        # Multi-depth search
        start_time = time.time()
        
        directions_and_qualities = []
        for depth in range(1, self.max_depth + 1):
            k = min(20 + depth * 5, 80)
            indices, scores = self.index.search(query_vec, k=k, mode='hybrid')
            direction, quality = self._message_passing_with_quality(indices.tolist(), depth)
            directions_and_qualities.append((direction, quality, depth))
        
        search_time = (time.time() - start_time) * 1000
        self.search_times.append(search_time)
        
        # Natural selection based on quality
        qualities = [q for _, q, _ in directions_and_qualities]
        total_quality = sum(qualities) + 0.001
        weights = [q / total_quality for q in qualities]
        
        self.depth_weights.append(weights)
        
        # Combine directions
        final_direction = np.zeros(7)
        for i, (direction, quality, depth) in enumerate(directions_and_qualities):
            final_direction += direction * weights[i]
            contribution = np.linalg.norm(direction * weights[i])
            self.hop_contributions[f'{depth}-hop'].append(contribution)
        
        # Normalize
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
            
            # Feature-based score
            feature_score = final_direction[2+i] if final_direction[2+i] > 0 else 0
            
            # Visit count influence - avoid highly visited directions
            next_pos = (x + dx, y + dy)
            if 0 <= next_pos[0] < self.height and 0 <= next_pos[1] < self.width:
                next_visits = self.visit_counts.get(next_pos, 0)
                visit_penalty = np.log1p(next_visits) / 20.0
            else:
                visit_penalty = 0
            
            action_scores[i] = max(0, 
                position_score * 0.5 + 
                feature_score * 0.4 - 
                visit_penalty * 0.1
            )
        
        # Minimal noise
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
            
            # Store success
            if len(self.index.metadata) > 0:
                self.index.metadata[-1]['action'] = action
                self.index.metadata[-1]['success'] = True
            
            self.position = (new_x, new_y)
            self.path.append(self.position)
            return True
        else:
            # Store failure
            if len(self.index.metadata) > 0:
                self.index.metadata[-1]['action'] = action
                self.index.metadata[-1]['success'] = False
                self.index.metadata[-1]['hit_wall'] = True
            return False
    
    def navigate(self, max_steps: int = 5000) -> Dict:
        """Navigate using visit count memory"""
        start_time = time.time()
        wall_hits = 0
        
        for step in range(max_steps):
            if self.position == self.goal:
                total_time = time.time() - start_time
                stats = self.index.get_statistics()
                
                print(f"\n🎉 SUCCESS with visit count memory!")
                
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
                hit_rate = wall_hits/step*100
                dist = abs(self.position[0]-self.goal[0])+abs(self.position[1]-self.goal[1])
                max_visits = max(self.visit_counts.values()) if self.visit_counts else 0
                
                print(f"Step {step}: pos={self.position}, dist={dist}, "
                      f"wall_hits={wall_hits} ({hit_rate:.1f}%), "
                      f"max_visits={max_visits}")
                
                if step % 500 == 0:
                    # Show top visited positions
                    top_visits = sorted(self.visit_counts.items(), 
                                      key=lambda x: x[1], reverse=True)[:3]
                    print(f"  Most visited: {top_visits}")
        
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
            'visit_counts': self.visit_counts,
            'index_stats': stats,
            'wall_hits': wall_hits
        }