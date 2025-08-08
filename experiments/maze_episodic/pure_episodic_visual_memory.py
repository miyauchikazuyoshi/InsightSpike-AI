#!/usr/bin/env python3
"""
Pure Episodic Navigator with Visual Episode Memory
- Adds 4 directional visual episodes after each move
- Query: (x, y, null, null, path, goal)
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add path for integrated index
sys.path.append(os.path.join(os.path.dirname(__file__), '../maze-optimized-search/src'))
from gedig_aware_integrated_index import GeDIGAwareIntegratedIndex


class PureEpisodicVisualMemory:
    """Navigator with visual episode memory"""
    
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
                'similarity_threshold': 0.6,
                'gedig_threshold': 0.5,
                'gedig_weight': 0.4,
                'max_edges_per_node': 10
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
        
        # Direction indices for vector encoding
        self.direction_indices = {
            'up': 2,
            'right': 3,
            'down': 4,
            'left': 5
        }
        
        # Multi-hop statistics
        self.hop_selections = {
            '1-hop': 0,
            '2-hop': 0,
            '3-hop': 0,
            'combined': 0
        }
        
        # Visual episode count
        self.visual_episodes = 0
    
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
        """Update visual memory for current position"""
        x, y = self.position
        if (x, y) not in self.visual_memory:
            self.visual_memory[(x, y)] = {}
            
            for action, (dx, dy) in self.action_deltas.items():
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.height and 0 <= ny < self.width:
                    self.visual_memory[(x, y)][action] = 'path' if self.maze[nx, ny] == 0 else 'wall'
    
    def _add_visual_episodes(self):
        """Add 4 directional visual episodes after movement"""
        x, y = self.position
        
        # Ensure visual memory is updated
        self._update_visual_memory()
        vis = self.visual_memory.get((x, y), {})
        
        # Add episode for each direction
        for action in self.actions:
            # Create visual episode vector: (x, y, direction, null, wall/path, null)
            vec = np.zeros(6, dtype=np.float32)
            
            # Position
            vec[0] = x / self.height
            vec[1] = y / self.width
            
            # Direction (one-hot encoding in dimensions 2-5)
            direction_idx = self.direction_indices[action]
            vec[direction_idx] = 1.0
            
            # Visual observation (dimension 4)
            # Dimension 3 is null (movement result)
            # Dimension 4 is visual observation
            # Dimension 5 is null (goal marker)
            if vis.get(action) == 'path':
                vec[4] = 1.0  # Path observed
            elif vis.get(action) == 'wall':
                vec[4] = -1.0  # Wall observed
            
            # Add visual episode with high confidence
            self.index.add_episode({
                'vec': vec,
                'text': f"Visual: {action} from ({x}, {y}) sees {vis.get(action, 'unknown')}",
                'pos': (x, y),
                'c_value': 0.9,  # High confidence for direct observation
                'type': 'visual',
                'direction': action,
                'observation': vis.get(action, 'unknown')
            })
            
            self.visual_episodes += 1
    
    def _create_movement_episode_vector(self, x: int, y: int, action: str, success: bool) -> np.ndarray:
        """Create movement episode vector: (x, y, direction, success, null, null)"""
        vec = np.zeros(6, dtype=np.float32)
        
        # Position
        vec[0] = x / self.height
        vec[1] = y / self.width
        
        # Direction (one-hot in dimensions 2-5)
        if action in self.direction_indices:
            vec[self.direction_indices[action]] = 1.0
        
        # Movement result (dimension 3)
        vec[3] = 1.0 if success else -1.0
        
        # Dimensions 4 and 5 are null for movement episodes
        
        return vec
    
    def _create_query_vector(self, x: int, y: int) -> np.ndarray:
        """Create query vector: (x, y, null, null, path, goal)"""
        vec = np.zeros(6, dtype=np.float32)
        
        # Current position
        vec[0] = x / self.height
        vec[1] = y / self.width
        
        # Dimensions 2-3 are null (no specific direction or result)
        
        # Dimension 4: Want to find paths
        vec[4] = 1.0  # Looking for episodes with paths
        
        # Dimension 5: Goal bias
        # Could enhance with goal direction
        goal_dx = self.goal[0] - x
        goal_dy = self.goal[1] - y
        goal_dist = abs(goal_dx) + abs(goal_dy)
        vec[5] = 1.0 / (goal_dist + 1)  # Inverse distance as goal proximity
        
        return vec
    
    def _message_passing_with_gedig(self, indices: List[int], depth: int) -> np.ndarray:
        """Message passing leveraging visual episodes"""
        if depth <= 0 or not indices:
            return np.zeros(6)
        
        # Initialize messages
        messages = {}
        for i, idx in enumerate(indices):
            messages[idx] = 1.0 / (i + 1)
        
        # Propagate through graph
        for d in range(depth):
            new_messages = {}
            
            for node, value in messages.items():
                if node not in self.index.graph:
                    continue
                    
                # Self-loop
                if node in new_messages:
                    new_messages[node] = max(new_messages[node], value * 0.8)
                else:
                    new_messages[node] = value * 0.8
                
                # Propagate to neighbors
                for neighbor in self.index.graph.neighbors(node):
                    edge_data = self.index.graph[node][neighbor]
                    
                    weight = edge_data.get('weight', 0.5)
                    gedig = edge_data.get('gedig', 1.0)
                    
                    propagation = value * weight * (1.0 - gedig * 0.5)
                    
                    if neighbor in new_messages:
                        new_messages[neighbor] = max(new_messages[neighbor], propagation)
                    else:
                        new_messages[neighbor] = propagation
            
            messages = new_messages
            if not messages:
                break
        
        # Aggregate with emphasis on visual episodes
        direction = np.zeros(6)
        total_weight = 0
        
        for idx, value in messages.items():
            if idx < len(self.index.metadata):
                episode = self.index.metadata[idx]
                vec = episode['vec']
                
                # Higher weight for visual episodes
                if episode.get('type') == 'visual':
                    type_weight = 2.0
                    # Extra weight if it shows a path
                    if episode.get('observation') == 'path':
                        type_weight *= 2.0
                else:
                    type_weight = 1.0
                
                # Success weight for movement episodes
                if 'success' in episode:
                    success_weight = 2.0 if episode['success'] else 0.3
                else:
                    success_weight = 1.0
                
                weight = value * type_weight * success_weight
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
            
        last_10 = self.recent_positions[-10:]
        unique_positions = len(set(last_10))
        
        return unique_positions < 4
    
    def get_action(self) -> Optional[str]:
        """Get next action using visual episode memory"""
        x, y = self.position
        query_vec = self._create_query_vector(x, y)
        
        # Track position
        self.recent_positions.append((x, y))
        if len(self.recent_positions) > 50:
            self.recent_positions.pop(0)
        
        # Check for loops
        if self._detect_loop():
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        # Multi-mode search
        start_time = time.time()
        
        indices_vec, scores_vec = self.index.search(query_vec, k=20, mode='vector')
        indices_gedig, scores_gedig = self.index.search(query_vec, k=20, mode='gedig')
        indices_hybrid, scores_hybrid = self.index.search(query_vec, k=30, mode='hybrid')
        
        search_time = (time.time() - start_time) * 1000
        self.search_times.append(search_time)
        
        # Message passing
        direction_1hop = self._message_passing_with_gedig(indices_vec.tolist(), 1)
        direction_2hop = self._message_passing_with_gedig(indices_hybrid.tolist(), 2)
        direction_3hop = self._message_passing_with_gedig(indices_gedig.tolist(), 3)
        
        # Adaptive selection
        if self.stuck_counter > 5:
            direction = direction_3hop
            self.hop_selections['3-hop'] += 1
        elif np.linalg.norm(direction_1hop) > 0.6:
            direction = direction_1hop
            self.hop_selections['1-hop'] += 1
        elif np.linalg.norm(direction_2hop) > 0.4:
            direction = direction_2hop
            self.hop_selections['2-hop'] += 1
        else:
            # Combined
            direction = direction_1hop * 0.3 + direction_2hop * 0.4 + direction_3hop * 0.3
            self.hop_selections['combined'] += 1
        
        # Convert direction to action scores
        action_scores = np.zeros(4)
        
        # Use direction vector to score actions
        # Dimensions 2-5 contain directional preferences from visual episodes
        for i, action in enumerate(self.actions):
            # Direction preference from visual episodes
            direction_idx = self.direction_indices[action]
            direction_score = direction[direction_idx]
            
            # Path preference (dimension 4 indicates paths seen)
            path_score = direction[4] if direction[4] > 0 else 0
            
            # Goal bias
            dx, dy = self.action_deltas[action]
            goal_dx = self.goal[0] - x
            goal_dy = self.goal[1] - y
            goal_alignment = (dx * np.sign(goal_dx) + dy * np.sign(goal_dy)) / 2
            
            action_scores[i] = max(0, 
                direction_score * 0.5 +  # Visual direction preference
                path_score * 0.3 +       # Path availability
                goal_alignment * 0.2     # Goal direction
            )
        
        # Add exploration noise
        if self.stuck_counter > 0:
            action_scores += np.random.random(4) * 0.1 * self.stuck_counter
        else:
            action_scores += np.random.random(4) * 0.01
        
        # Select action
        if np.sum(action_scores) > 0:
            probs = action_scores / np.sum(action_scores)
            return np.random.choice(self.actions, p=probs)
        
        return np.random.choice(self.actions)
    
    def move(self, action: str) -> bool:
        """Execute action and add movement/visual episodes"""
        if action not in self.actions:
            return False
            
        dx, dy = self.action_deltas[action]
        new_x, new_y = self.position[0] + dx, self.position[1] + dy
        
        # Try to move
        if (0 <= new_x < self.height and 0 <= new_y < self.width and 
            self.maze[new_x, new_y] == 0):
            
            # Add movement episode (before moving)
            movement_vec = self._create_movement_episode_vector(
                self.position[0], self.position[1], action, True
            )
            self.index.add_episode({
                'vec': movement_vec,
                'text': f"Moved {action} from {self.position} successfully",
                'pos': self.position,
                'c_value': 0.8,
                'type': 'movement',
                'action': action,
                'success': True
            })
            
            # Actually move
            self.position = (new_x, new_y)
            self.path.append(self.position)
            
            # Add visual episodes from new position
            self._add_visual_episodes()
            
            return True
        else:
            # Add failed movement episode
            movement_vec = self._create_movement_episode_vector(
                self.position[0], self.position[1], action, False
            )
            self.index.add_episode({
                'vec': movement_vec,
                'text': f"Failed to move {action} from {self.position}",
                'pos': self.position,
                'c_value': 0.6,
                'type': 'movement',
                'action': action,
                'success': False
            })
            
            return False
    
    def navigate(self, max_steps: int = 5000) -> Dict:
        """Navigate maze using visual episode memory"""
        start_time = time.time()
        wall_hits = 0
        
        # Add initial visual episodes
        self._add_visual_episodes()
        
        for step in range(max_steps):
            if self.position == self.goal:
                total_time = time.time() - start_time
                stats = self.index.get_statistics()
                
                print(f"\n🎉 SUCCESS! Reached goal in {step} steps!")
                print(f"Visual episodes created: {self.visual_episodes}")
                
                return {
                    'success': True,
                    'steps': step,
                    'total_episodes': len(self.index.metadata),
                    'visual_episodes': self.visual_episodes,
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
                hit_rate = wall_hits/step*100
                dist = abs(self.position[0]-self.goal[0])+abs(self.position[1]-self.goal[1])
                
                print(f"Step {step}: pos={self.position}, dist={dist}, "
                      f"episodes={stats['episodes']} (visual: {self.visual_episodes}), "
                      f"wall_hits={wall_hits} ({hit_rate:.1f}%)")
        
        total_time = time.time() - start_time
        stats = self.index.get_statistics()
        
        return {
            'success': False,
            'steps': max_steps,
            'total_episodes': len(self.index.metadata),
            'visual_episodes': self.visual_episodes,
            'total_time': total_time,
            'avg_search_time': np.mean(self.search_times) if self.search_times else 0,
            'search_times': self.search_times,
            'path': self.path,
            'hop_selections': self.hop_selections,
            'index_stats': stats,
            'wall_hits': wall_hits
        }