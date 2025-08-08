#!/usr/bin/env python3
"""
Pure Episodic Navigator with aspirational query and improved multi-hop
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add path for integrated index
sys.path.append(os.path.join(os.path.dirname(__file__), '../maze-optimized-search/src'))
from gedig_aware_integrated_index import GeDIGAwareIntegratedIndex


class PureEpisodicGeDIGQueryMultihop:
    """Navigator with aspirational query and better multi-hop utilization"""
    
    def __init__(self, maze: np.ndarray, message_depth: int = 3):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = self._find_start()
        self.goal = self._find_goal()
        self.message_depth = message_depth
        
        # geDIG-aware integrated index
        self.index = GeDIGAwareIntegratedIndex(
            dimension=6,
            config={
                'similarity_threshold': 0.5,  # Lower threshold for more connections
                'gedig_threshold': 0.6,
                'gedig_weight': 0.3,  # Less weight on geDIG for more exploration
                'max_edges_per_node': 12  # More edges for better multi-hop
            }
        )
        
        # Memory systems
        self.visual_memory = {}
        self.search_times = []
        self.path = [(self.position[0], self.position[1])]
        self.stuck_counter = 0
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
        self.hop_selections = {
            '1-hop': 0,
            '2-hop': 0,
            '3-hop': 0,
            'combined': 0
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
    
    def _update_visual_memory(self):
        x, y = self.position
        if (x, y) not in self.visual_memory:
            self.visual_memory[(x, y)] = {}
            
            for action, (dx, dy) in self.action_deltas.items():
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.height and 0 <= ny < self.width:
                    self.visual_memory[(x, y)][action] = 'path' if self.maze[nx, ny] == 0 else 'wall'
    
    def _create_episode_vector(self, x: int, y: int) -> np.ndarray:
        """Create episode vector for STORAGE (includes actual wall info)"""
        vec = np.zeros(6, dtype=np.float32)
        
        # Normalized position
        vec[0] = x / self.height
        vec[1] = y / self.width
        
        # Visual features - actual walls
        vis = self.visual_memory.get((x, y), {})
        for i, action in enumerate(self.actions):
            if vis.get(action) == 'path':
                vec[2+i] = 1.0
            elif vis.get(action) == 'wall':
                vec[2+i] = -1.0
        
        return vec
    
    def _create_query_vector(self, x: int, y: int) -> np.ndarray:
        """Create query vector for SEARCH with goal bias"""
        vec = np.zeros(6, dtype=np.float32)
        
        # Current position
        vec[0] = x / self.height
        vec[1] = y / self.width
        
        # Goal-biased aspirational features
        goal_dx = self.goal[0] - x
        goal_dy = self.goal[1] - y
        
        # Bias toward goal direction
        if abs(goal_dx) > abs(goal_dy):
            if goal_dx < 0:
                vec[2] = 2.0  # Strong preference for up
            else:
                vec[3] = 2.0  # Strong preference for down
        else:
            if goal_dy < 0:
                vec[4] = 2.0  # Strong preference for left
            else:
                vec[5] = 2.0  # Strong preference for right
        
        # Moderate preference for other directions
        for i in range(4):
            if vec[2+i] == 0:
                vec[2+i] = 0.5  # Some openness in all directions
        
        return vec
    
    def _message_passing_with_gedig(self, indices: List[int], depth: int) -> np.ndarray:
        """Enhanced message passing with better propagation"""
        if depth <= 0 or not indices:
            return np.zeros(6)
        
        # Initialize messages with distance-based weights
        messages = {}
        for i, idx in enumerate(indices):
            # Give higher initial weight to closer episodes
            messages[idx] = 1.0 / (i + 1)
        
        # Propagate through graph
        for d in range(depth):
            new_messages = {}
            
            for node, value in messages.items():
                if node not in self.index.graph:
                    continue
                    
                # Add self-loop to maintain information
                if node in new_messages:
                    new_messages[node] = max(new_messages[node], value * 0.8)
                else:
                    new_messages[node] = value * 0.8
                
                # Propagate to neighbors
                for neighbor in self.index.graph.neighbors(node):
                    edge_data = self.index.graph[node][neighbor]
                    
                    # Use combined score
                    weight = edge_data.get('weight', 0.5)
                    gedig = edge_data.get('gedig', 1.0)
                    
                    # Balance between similarity and structural difference
                    propagation = value * weight * (1.0 - gedig * 0.5)
                    
                    if neighbor in new_messages:
                        new_messages[neighbor] = max(new_messages[neighbor], propagation)
                    else:
                        new_messages[neighbor] = propagation
            
            messages = new_messages
            if not messages:
                break
        
        # Aggregate into direction vector
        direction = np.zeros(6)
        total_weight = 0
        
        for idx, value in messages.items():
            if idx < len(self.index.metadata):
                episode = self.index.metadata[idx]
                vec = episode['vec']
                
                # Weight by success and recency
                success_weight = 2.0 if episode.get('success', False) else 0.5
                recency_weight = 1.0 / (len(self.index.metadata) - idx + 1) ** 0.5
                
                weight = value * success_weight * recency_weight
                direction += vec * weight
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            direction = direction / total_weight
            
        return direction
    
    def _detect_loop(self) -> bool:
        """Detect if stuck in a loop"""
        if len(self.recent_positions) < 20:
            return False
            
        # Check if we're revisiting same positions
        last_10 = self.recent_positions[-10:]
        unique_positions = len(set(last_10))
        
        return unique_positions < 4
    
    def get_action(self) -> Optional[str]:
        """Get next action using multi-hop reasoning"""
        self._update_visual_memory()
        
        x, y = self.position
        current_vec = self._create_episode_vector(x, y)  # Actual situation
        query_vec = self._create_query_vector(x, y)      # Goal-biased query
        
        # Track position
        self.recent_positions.append((x, y))
        if len(self.recent_positions) > 50:
            self.recent_positions.pop(0)
        
        # Check for loops
        if self._detect_loop():
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        # Store episode
        base_confidence = 0.7
        if self.stuck_counter > 5:
            base_confidence = 0.3
        
        episode_idx = self.index.add_episode({
            'vec': current_vec,
            'text': f"Episode at ({x}, {y})",
            'pos': (x, y),
            'c_value': base_confidence,
            'stuck': self.stuck_counter > 0,
            'distance_to_goal': abs(x - self.goal[0]) + abs(y - self.goal[1])
        })
        
        # Multi-mode search with different k values
        start_time = time.time()
        
        # Use different search strategies
        indices_vec, scores_vec = self.index.search(query_vec, k=10, mode='vector')
        indices_gedig, scores_gedig = self.index.search(query_vec, k=20, mode='gedig')
        indices_hybrid, scores_hybrid = self.index.search(query_vec, k=30, mode='hybrid')
        
        search_time = (time.time() - start_time) * 1000
        self.search_times.append(search_time)
        
        # Multi-hop message passing
        direction_1hop = self._message_passing_with_gedig(indices_vec.tolist(), 1)
        direction_2hop = self._message_passing_with_gedig(indices_hybrid.tolist(), 2)
        direction_3hop = self._message_passing_with_gedig(indices_gedig.tolist(), 3)
        
        # Combine directions intelligently
        if self.stuck_counter > 10:
            # Deep exploration when very stuck
            direction = direction_3hop * 0.6 + direction_2hop * 0.3 + direction_1hop * 0.1
            self.hop_selections['3-hop'] += 1
        elif self.stuck_counter > 5:
            # Medium exploration when stuck
            direction = direction_2hop * 0.6 + direction_3hop * 0.3 + direction_1hop * 0.1
            self.hop_selections['2-hop'] += 1
        elif np.linalg.norm(direction_1hop) > 0.3:
            # Use 1-hop when confident
            if np.linalg.norm(direction_2hop) > 0.2:
                # Combine if 2-hop also has signal
                direction = direction_1hop * 0.7 + direction_2hop * 0.3
                self.hop_selections['combined'] += 1
            else:
                direction = direction_1hop
                self.hop_selections['1-hop'] += 1
        else:
            # Default to multi-hop combination
            direction = direction_1hop * 0.2 + direction_2hop * 0.5 + direction_3hop * 0.3
            self.hop_selections['combined'] += 1
        
        # Convert direction to action scores
        action_scores = np.zeros(4)
        
        # Project direction onto ALL actions (no wall filtering!)
        for i, action in enumerate(self.actions):
            dx, dy = self.action_deltas[action]
            # Create action vector
            action_vec = np.zeros(6)
            action_vec[0] = dx / self.height
            action_vec[1] = dy / self.width
            
            # Score based on alignment with direction
            position_score = np.dot(direction[:2], action_vec[:2])
            
            # Also consider the path features
            feature_score = direction[2+i] if direction[2+i] > 0 else 0
            
            action_scores[i] = max(0, position_score + feature_score * 0.3)
        
        # Add small random noise to break ties
        action_scores += np.random.random(4) * 0.01
        
        # Select action based purely on scores
        if np.sum(action_scores) > 0:
            probs = action_scores / np.sum(action_scores)
            return np.random.choice(self.actions, p=probs)
        
        # Fallback: random action
        return np.random.choice(self.actions)
    
    def move(self, action: str) -> bool:
        """Execute action and store result"""
        if action not in self.actions:
            return False
            
        dx, dy = self.action_deltas[action]
        new_x, new_y = self.position[0] + dx, self.position[1] + dy
        
        # Physical constraints only
        if (0 <= new_x < self.height and 0 <= new_y < self.width and 
            self.maze[new_x, new_y] == 0):
            
            # Store successful action
            if len(self.index.metadata) > 0:
                self.index.metadata[-1]['action'] = action
                self.index.metadata[-1]['success'] = True
            
            self.position = (new_x, new_y)
            self.path.append(self.position)
            return True
        else:
            # Store failed action
            if len(self.index.metadata) > 0:
                self.index.metadata[-1]['action'] = action
                self.index.metadata[-1]['success'] = False
            return False
    
    def navigate(self, max_steps: int = 5000) -> Dict:
        """Navigate maze and return results"""
        start_time = time.time()
        wall_hits = 0
        
        for step in range(max_steps):
            if self.position == self.goal:
                total_time = time.time() - start_time
                stats = self.index.get_statistics()
                
                return {
                    'success': True,
                    'steps': step,
                    'total_episodes': len(self.index.metadata),
                    'total_time': total_time,
                    'avg_search_time': np.mean(self.search_times) if self.search_times else 0,
                    'search_times': self.search_times,
                    'path': self.path,
                    'hop_selections': self.hop_selections,
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
                print(f"Step {step}: pos={self.position}, episodes={stats['episodes']}, "
                      f"wall_hits={wall_hits}, hit_rate={wall_hits/step*100:.1f}%, "
                      f"dist_to_goal={abs(self.position[0]-self.goal[0])+abs(self.position[1]-self.goal[1])}")
        
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
            'hop_selections': self.hop_selections,
            'index_stats': stats,
            'wall_hits': wall_hits
        }