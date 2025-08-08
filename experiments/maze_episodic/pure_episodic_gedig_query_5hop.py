#!/usr/bin/env python3
"""
Pure Episodic Navigator with 5-hop deep message passing
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add path for integrated index
sys.path.append(os.path.join(os.path.dirname(__file__), '../maze-optimized-search/src'))
from gedig_aware_integrated_index import GeDIGAwareIntegratedIndex


class PureEpisodicGeDIGQuery5Hop:
    """Navigator with up to 5-hop message passing"""
    
    def __init__(self, maze: np.ndarray, message_depth: int = 5):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = self._find_start()
        self.goal = self._find_goal()
        self.message_depth = message_depth
        
        # geDIG-aware integrated index with more connections
        self.index = GeDIGAwareIntegratedIndex(
            dimension=6,
            config={
                'similarity_threshold': 0.4,  # Lower for more connections
                'gedig_threshold': 0.7,
                'gedig_weight': 0.25,  # Balanced weight
                'max_edges_per_node': 20  # More edges for deeper propagation
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
        
        # Extended multi-hop statistics
        self.hop_selections = {
            '1-hop': 0,
            '2-hop': 0,
            '3-hop': 0,
            '4-hop': 0,
            '5-hop': 0,
            'adaptive': 0
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
        """Create episode vector for STORAGE"""
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
        """Create query vector with strong goal bias and path preference"""
        vec = np.zeros(6, dtype=np.float32)
        
        # Current position
        vec[0] = x / self.height
        vec[1] = y / self.width
        
        # Strong goal-directed bias
        goal_dx = self.goal[0] - x
        goal_dy = self.goal[1] - y
        goal_dist = abs(goal_dx) + abs(goal_dy)
        
        # Adaptive feature weights based on distance to goal
        if goal_dist > 10:
            # Far from goal - strong directional bias
            strength = 3.0
        else:
            # Near goal - moderate bias
            strength = 2.0
        
        # Set directional preferences
        if abs(goal_dx) > abs(goal_dy):
            if goal_dx < 0:
                vec[2] = strength  # Up
                vec[3] = 0.3      # Down (avoid opposite)
            else:
                vec[3] = strength  # Down
                vec[2] = 0.3      # Up (avoid opposite)
            vec[4] = 1.0  # Left (moderate)
            vec[5] = 1.0  # Right (moderate)
        else:
            if goal_dy < 0:
                vec[4] = strength  # Left
                vec[5] = 0.3      # Right (avoid opposite)
            else:
                vec[5] = strength  # Right
                vec[4] = 0.3      # Left (avoid opposite)
            vec[2] = 1.0  # Up (moderate)
            vec[3] = 1.0  # Down (moderate)
        
        return vec
    
    def _message_passing_enhanced(self, indices: List[int], depth: int) -> np.ndarray:
        """Enhanced message passing with exponential decay and path weighting"""
        if depth <= 0 or not indices:
            return np.zeros(6)
        
        # Initialize with stronger initial weights
        messages = {}
        for i, idx in enumerate(indices):
            # Exponential decay for initial weights
            messages[idx] = np.exp(-i * 0.1)
        
        # Track visited nodes to avoid cycles
        all_visited = set(indices)
        
        # Propagate through graph
        for d in range(depth):
            new_messages = {}
            decay_factor = 0.7 ** d  # Exponential decay per hop
            
            for node, value in messages.items():
                if node not in self.index.graph:
                    continue
                
                # Self-loop with decay
                if d < depth - 1:  # Don't self-loop on last iteration
                    new_messages[node] = value * 0.5 * decay_factor
                
                # Get neighbors sorted by edge weight
                neighbors = []
                for neighbor in self.index.graph.neighbors(node):
                    edge_data = self.index.graph[node][neighbor]
                    weight = edge_data.get('weight', 0.5)
                    gedig = edge_data.get('gedig', 1.0)
                    combined_score = weight * (1.0 - gedig * 0.3)
                    neighbors.append((neighbor, combined_score))
                
                # Sort by score and propagate to best neighbors
                neighbors.sort(key=lambda x: x[1], reverse=True)
                for neighbor, score in neighbors[:10]:  # Top 10 neighbors
                    if neighbor in all_visited and d < depth - 1:
                        continue  # Avoid revisiting except on last hop
                    
                    propagation = value * score * decay_factor
                    
                    if neighbor in new_messages:
                        new_messages[neighbor] = max(new_messages[neighbor], propagation)
                    else:
                        new_messages[neighbor] = propagation
                    
                    all_visited.add(neighbor)
            
            messages = new_messages
            if not messages:
                break
        
        # Aggregate with path quality weighting
        direction = np.zeros(6)
        total_weight = 0
        
        for idx, value in messages.items():
            if idx < len(self.index.metadata):
                episode = self.index.metadata[idx]
                vec = episode['vec']
                
                # Multi-factor weighting
                success_weight = 3.0 if episode.get('success', False) else 0.3
                stuck_penalty = 0.1 if episode.get('stuck', False) else 1.0
                
                # Distance to goal bonus
                if 'distance_to_goal' in episode:
                    goal_weight = 1.0 / (episode['distance_to_goal'] + 1)
                else:
                    goal_weight = 0.5
                
                # Recency bonus
                recency = 1.0 / (len(self.index.metadata) - idx + 10) ** 0.3
                
                weight = value * success_weight * stuck_penalty * goal_weight * recency
                direction += vec * weight
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            direction = direction / total_weight
            
        return direction
    
    def _detect_loop(self) -> bool:
        """Enhanced loop detection"""
        if len(self.recent_positions) < 30:
            return False
            
        # Check multiple window sizes
        for window in [10, 20, 30]:
            if len(self.recent_positions) >= window:
                last_positions = self.recent_positions[-window:]
                unique_positions = len(set(last_positions))
                if unique_positions < window * 0.3:  # Less than 30% unique
                    return True
        
        return False
    
    def get_action(self) -> Optional[str]:
        """Get next action using deep multi-hop reasoning"""
        self._update_visual_memory()
        
        x, y = self.position
        current_vec = self._create_episode_vector(x, y)
        query_vec = self._create_query_vector(x, y)
        
        # Track position
        self.recent_positions.append((x, y))
        if len(self.recent_positions) > 100:
            self.recent_positions.pop(0)
        
        # Enhanced loop detection
        if self._detect_loop():
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)
        
        # Store episode with rich metadata
        base_confidence = 0.8
        if self.stuck_counter > 10:
            base_confidence = 0.2
        elif self.stuck_counter > 5:
            base_confidence = 0.4
        
        episode_idx = self.index.add_episode({
            'vec': current_vec,
            'text': f"Episode at ({x}, {y})",
            'pos': (x, y),
            'c_value': base_confidence,
            'stuck': self.stuck_counter > 0,
            'distance_to_goal': abs(x - self.goal[0]) + abs(y - self.goal[1]),
            'step': len(self.path)
        })
        
        # Adaptive search with different strategies
        start_time = time.time()
        
        # Search with increasing k values for deeper hops
        indices_1hop, _ = self.index.search(query_vec, k=20, mode='vector')
        indices_2hop, _ = self.index.search(query_vec, k=30, mode='hybrid')
        indices_3hop, _ = self.index.search(query_vec, k=40, mode='hybrid')
        indices_4hop, _ = self.index.search(query_vec, k=50, mode='gedig')
        indices_5hop, _ = self.index.search(query_vec, k=60, mode='gedig')
        
        search_time = (time.time() - start_time) * 1000
        self.search_times.append(search_time)
        
        # Deep message passing
        directions = []
        directions.append(self._message_passing_enhanced(indices_1hop.tolist(), 1))
        directions.append(self._message_passing_enhanced(indices_2hop.tolist(), 2))
        directions.append(self._message_passing_enhanced(indices_3hop.tolist(), 3))
        directions.append(self._message_passing_enhanced(indices_4hop.tolist(), 4))
        directions.append(self._message_passing_enhanced(indices_5hop.tolist(), 5))
        
        # Adaptive combination based on signal strength and situation
        strengths = [np.linalg.norm(d) for d in directions]
        
        if self.stuck_counter > 15:
            # Very stuck - use deepest propagation
            direction = directions[4]
            self.hop_selections['5-hop'] += 1
        elif self.stuck_counter > 10:
            # Stuck - combine 4 and 5 hop
            direction = directions[3] * 0.6 + directions[4] * 0.4
            self.hop_selections['4-hop'] += 1
        elif self.stuck_counter > 5:
            # Somewhat stuck - use 3-hop with some 4-hop
            direction = directions[2] * 0.7 + directions[3] * 0.3
            self.hop_selections['3-hop'] += 1
        elif strengths[0] > 0.5 and strengths[1] > 0.4:
            # Strong local signal - combine 1 and 2 hop
            direction = directions[0] * 0.6 + directions[1] * 0.4
            self.hop_selections['2-hop'] += 1
        elif strengths[0] > 0.6:
            # Very strong 1-hop signal
            direction = directions[0]
            self.hop_selections['1-hop'] += 1
        else:
            # Adaptive weighted combination
            total_strength = sum(strengths) + 0.001
            weights = [s / total_strength for s in strengths]
            direction = sum(d * w for d, w in zip(directions, weights))
            self.hop_selections['adaptive'] += 1
        
        # Convert to action scores with enhanced projection
        action_scores = np.zeros(4)
        
        for i, action in enumerate(self.actions):
            dx, dy = self.action_deltas[action]
            
            # Position-based score
            action_vec = np.zeros(2)
            action_vec[0] = dx / self.height
            action_vec[1] = dy / self.width
            position_score = np.dot(direction[:2], action_vec)
            
            # Feature-based score (path preference)
            feature_score = direction[2+i] if direction[2+i] > 0 else 0
            
            # Goal direction bonus
            goal_vec = np.array([self.goal[0] - x, self.goal[1] - y])
            goal_vec = goal_vec / (np.linalg.norm(goal_vec) + 0.001)
            goal_alignment = np.dot(goal_vec, [dx, dy])
            
            # Combined score
            action_scores[i] = max(0, 
                position_score * 0.5 + 
                feature_score * 0.3 + 
                goal_alignment * 0.2
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
                self.index.metadata[-1]['next_pos'] = (new_x, new_y)
            
            self.position = (new_x, new_y)
            self.path.append(self.position)
            return True
        else:
            # Store failed action
            if len(self.index.metadata) > 0:
                self.index.metadata[-1]['action'] = action
                self.index.metadata[-1]['success'] = False
                self.index.metadata[-1]['hit_wall'] = True
            return False
    
    def navigate(self, max_steps: int = 5000) -> Dict:
        """Navigate maze and return results"""
        start_time = time.time()
        wall_hits = 0
        
        for step in range(max_steps):
            if self.position == self.goal:
                total_time = time.time() - start_time
                stats = self.index.get_statistics()
                
                print(f"\n🎉 SUCCESS! Reached goal in {step} steps!")
                
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
                hit_rate = wall_hits/step*100
                dist = abs(self.position[0]-self.goal[0])+abs(self.position[1]-self.goal[1])
                
                print(f"Step {step}: pos={self.position}, dist={dist}, "
                      f"episodes={stats['episodes']}, edges={stats['edges']}, "
                      f"wall_hits={wall_hits} ({hit_rate:.1f}%)")
                
                # Show hop usage
                if step % 500 == 0:
                    print("  Hop usage:", {k: v for k, v in self.hop_selections.items() if v > 0})
        
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