#!/usr/bin/env python3
"""
Pure Episodic Navigator with 10-hop ultra-deep message passing
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add path for integrated index
sys.path.append(os.path.join(os.path.dirname(__file__), '../maze-optimized-search/src'))
from gedig_aware_integrated_index import GeDIGAwareIntegratedIndex


class PureEpisodicGeDIGQuery10Hop:
    """Navigator with up to 10-hop ultra-deep message passing"""
    
    def __init__(self, maze: np.ndarray, message_depth: int = 10):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = self._find_start()
        self.goal = self._find_goal()
        self.message_depth = message_depth
        
        # geDIG-aware integrated index with dense connections for deep propagation
        self.index = GeDIGAwareIntegratedIndex(
            dimension=6,
            config={
                'similarity_threshold': 0.3,  # Very low for maximum connectivity
                'gedig_threshold': 0.8,
                'gedig_weight': 0.2,  # Low weight for more exploration
                'max_edges_per_node': 30  # Many edges for deep propagation
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
        self.hop_selections = {f'{i}-hop': 0 for i in range(1, 11)}
        self.hop_selections['adaptive'] = 0
        
        # Deep propagation cache to avoid recomputation
        self.propagation_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
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
        """Create query vector with strong goal bias and adaptive features"""
        vec = np.zeros(6, dtype=np.float32)
        
        # Current position
        vec[0] = x / self.height
        vec[1] = y / self.width
        
        # Goal-directed features with distance-based adaptation
        goal_dx = self.goal[0] - x
        goal_dy = self.goal[1] - y
        goal_dist = abs(goal_dx) + abs(goal_dy)
        
        # Ultra-strong bias when far from goal
        if goal_dist > 20:
            strength = 4.0
        elif goal_dist > 10:
            strength = 3.0
        else:
            strength = 2.5
        
        # Set primary direction with strong preference
        primary_set = False
        if abs(goal_dx) > abs(goal_dy) * 1.5:  # Strong vertical preference
            if goal_dx < 0:
                vec[2] = strength  # Up
                vec[3] = 0.1      # Down (strongly avoid)
                primary_set = True
            else:
                vec[3] = strength  # Down
                vec[2] = 0.1      # Up (strongly avoid)
                primary_set = True
        elif abs(goal_dy) > abs(goal_dx) * 1.5:  # Strong horizontal preference
            if goal_dy < 0:
                vec[4] = strength  # Left
                vec[5] = 0.1      # Right (strongly avoid)
                primary_set = True
            else:
                vec[5] = strength  # Right
                vec[4] = 0.1      # Left (strongly avoid)
                primary_set = True
        
        # If no strong preference, use balanced approach
        if not primary_set:
            if goal_dx < 0:
                vec[2] = strength * 0.8  # Up
                vec[3] = 0.3
            else:
                vec[3] = strength * 0.8  # Down
                vec[2] = 0.3
            
            if goal_dy < 0:
                vec[4] = strength * 0.8  # Left
                vec[5] = 0.3
            else:
                vec[5] = strength * 0.8  # Right
                vec[4] = 0.3
        else:
            # Set secondary directions
            for i in range(4):
                if vec[2+i] == 0:
                    vec[2+i] = 0.8
        
        return vec
    
    def _message_passing_ultra_deep(self, indices: List[int], depth: int) -> np.ndarray:
        """Ultra-deep message passing with optimizations for 10-hop"""
        if depth <= 0 or not indices:
            return np.zeros(6)
        
        # Check cache
        cache_key = (tuple(indices[:10]), depth)  # Use first 10 indices as key
        if cache_key in self.propagation_cache:
            self.cache_hits += 1
            return self.propagation_cache[cache_key].copy()
        self.cache_misses += 1
        
        # Initialize with exponential decay
        messages = {}
        for i, idx in enumerate(indices[:40]):  # Limit initial nodes for efficiency
            messages[idx] = np.exp(-i * 0.05)  # Slower decay for deep propagation
        
        # Track all visited nodes
        all_visited = set(indices)
        
        # Deep propagation with optimizations
        for d in range(depth):
            new_messages = {}
            
            # Adaptive decay based on depth
            if d < 3:
                decay_factor = 0.85 ** d
            elif d < 6:
                decay_factor = 0.75 ** (d - 3) * 0.614  # 0.85^3
            else:
                decay_factor = 0.65 ** (d - 6) * 0.316  # 0.85^3 * 0.75^3
            
            # Limit nodes to process at deeper levels
            if d > 5:
                # Sort by value and take top nodes
                sorted_nodes = sorted(messages.items(), key=lambda x: x[1], reverse=True)
                messages = dict(sorted_nodes[:100])  # Process only top 100 nodes
            
            for node, value in messages.items():
                if node not in self.index.graph:
                    continue
                
                # Self-loop with strong decay at deeper levels
                if d < depth - 1:
                    self_decay = 0.3 if d < 5 else 0.1
                    new_messages[node] = value * self_decay * decay_factor
                
                # Get and sort neighbors
                neighbors = []
                for neighbor in self.index.graph.neighbors(node):
                    if d < 5 or neighbor not in all_visited:  # Allow revisits at deeper levels
                        edge_data = self.index.graph[node][neighbor]
                        weight = edge_data.get('weight', 0.5)
                        gedig = edge_data.get('gedig', 1.0)
                        
                        # Adaptive edge scoring based on depth
                        if d < 3:
                            combined_score = weight * (1.0 - gedig * 0.2)
                        else:
                            combined_score = weight * (1.0 - gedig * 0.4)  # More exploration
                        
                        neighbors.append((neighbor, combined_score))
                
                # Sort and limit neighbors based on depth
                neighbors.sort(key=lambda x: x[1], reverse=True)
                max_neighbors = 15 if d < 5 else 8
                
                for neighbor, score in neighbors[:max_neighbors]:
                    propagation = value * score * decay_factor
                    
                    if neighbor in new_messages:
                        new_messages[neighbor] = max(new_messages[neighbor], propagation)
                    else:
                        new_messages[neighbor] = propagation
                    
                    all_visited.add(neighbor)
            
            messages = new_messages
            if not messages or (d > 5 and len(messages) < 10):
                break  # Early stopping if propagation dies out
        
        # Enhanced aggregation with richer weighting
        direction = np.zeros(6)
        total_weight = 0
        
        for idx, value in messages.items():
            if idx < len(self.index.metadata):
                episode = self.index.metadata[idx]
                vec = episode['vec']
                
                # Multi-factor weighting
                success_weight = 5.0 if episode.get('success', False) else 0.2
                stuck_penalty = 0.05 if episode.get('stuck', False) else 1.0
                
                # Goal proximity bonus
                if 'distance_to_goal' in episode:
                    goal_weight = np.exp(-episode['distance_to_goal'] * 0.1)
                else:
                    goal_weight = 0.3
                
                # Wall hit penalty
                wall_penalty = 0.5 if episode.get('hit_wall', False) else 1.0
                
                # Recency with slower decay
                recency = 1.0 / (len(self.index.metadata) - idx + 50) ** 0.2
                
                weight = value * success_weight * stuck_penalty * goal_weight * wall_penalty * recency
                direction += vec * weight
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            direction = direction / total_weight
        
        # Cache result
        self.propagation_cache[cache_key] = direction.copy()
        
        # Clean cache if too large
        if len(self.propagation_cache) > 100:
            # Remove oldest entries
            keys = list(self.propagation_cache.keys())
            for key in keys[:20]:
                del self.propagation_cache[key]
            
        return direction
    
    def _detect_loop(self) -> bool:
        """Multi-scale loop detection"""
        if len(self.recent_positions) < 40:
            return False
            
        # Check multiple time windows
        for window in [10, 20, 30, 40]:
            if len(self.recent_positions) >= window:
                last_positions = self.recent_positions[-window:]
                unique_positions = len(set(last_positions))
                if unique_positions < window * 0.25:  # Less than 25% unique
                    return True
        
        # Check for oscillation patterns
        if len(self.recent_positions) >= 20:
            last_20 = self.recent_positions[-20:]
            # Check if bouncing between few positions
            position_counts = {}
            for pos in last_20:
                position_counts[pos] = position_counts.get(pos, 0) + 1
            if max(position_counts.values()) > 8:  # Same position > 8 times
                return True
        
        return False
    
    def get_action(self) -> Optional[str]:
        """Get next action using ultra-deep multi-hop reasoning"""
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
            self.stuck_counter = min(self.stuck_counter + 2, 30)  # Faster increment
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)
        
        # Store episode with detailed metadata
        base_confidence = 0.9
        if self.stuck_counter > 20:
            base_confidence = 0.1
        elif self.stuck_counter > 10:
            base_confidence = 0.3
        elif self.stuck_counter > 5:
            base_confidence = 0.5
        
        episode_idx = self.index.add_episode({
            'vec': current_vec,
            'text': f"Episode at ({x}, {y})",
            'pos': (x, y),
            'c_value': base_confidence,
            'stuck': self.stuck_counter > 0,
            'stuck_level': self.stuck_counter,
            'distance_to_goal': abs(x - self.goal[0]) + abs(y - self.goal[1]),
            'step': len(self.path)
        })
        
        # Adaptive search with increasing k for deeper hops
        start_time = time.time()
        
        # Search with exponentially increasing k
        k_values = [20, 30, 40, 60, 80, 100, 120, 140, 160, 180]
        search_modes = ['vector', 'hybrid', 'hybrid', 'hybrid', 'gedig', 
                       'gedig', 'gedig', 'gedig', 'gedig', 'gedig']
        
        all_indices = []
        for i in range(min(10, self.stuck_counter // 3 + 3)):  # Adaptive depth
            indices, _ = self.index.search(query_vec, k=k_values[i], mode=search_modes[i])
            all_indices.append(indices)
        
        search_time = (time.time() - start_time) * 1000
        self.search_times.append(search_time)
        
        # Ultra-deep message passing
        directions = []
        for i, indices in enumerate(all_indices):
            depth = i + 1
            direction = self._message_passing_ultra_deep(indices.tolist(), depth)
            directions.append(direction)
        
        # Adaptive combination based on situation
        if self.stuck_counter > 25:
            # Ultra stuck - use deepest available
            if len(directions) >= 10:
                direction = directions[9]
                self.hop_selections['10-hop'] += 1
            elif len(directions) >= 8:
                direction = directions[7] * 0.7 + directions[6] * 0.3
                self.hop_selections['8-hop'] += 1
            else:
                direction = directions[-1]
                self.hop_selections[f'{len(directions)}-hop'] += 1
        elif self.stuck_counter > 20:
            # Very stuck - 7-9 hop
            if len(directions) >= 9:
                direction = directions[8] * 0.5 + directions[7] * 0.3 + directions[6] * 0.2
                self.hop_selections['9-hop'] += 1
            elif len(directions) >= 7:
                direction = directions[6] * 0.6 + directions[5] * 0.4
                self.hop_selections['7-hop'] += 1
            else:
                direction = directions[-1]
                self.hop_selections[f'{len(directions)}-hop'] += 1
        elif self.stuck_counter > 15:
            # Stuck - 5-7 hop
            if len(directions) >= 6:
                direction = directions[5] * 0.5 + directions[4] * 0.3 + directions[3] * 0.2
                self.hop_selections['6-hop'] += 1
            else:
                direction = directions[-1]
                self.hop_selections[f'{len(directions)}-hop'] += 1
        elif self.stuck_counter > 10:
            # Somewhat stuck - 4-5 hop
            if len(directions) >= 5:
                direction = directions[4] * 0.6 + directions[3] * 0.4
                self.hop_selections['5-hop'] += 1
            else:
                direction = directions[-1]
                self.hop_selections[f'{len(directions)}-hop'] += 1
        else:
            # Not stuck - adaptive based on signal strength
            strengths = [np.linalg.norm(d) for d in directions[:5]]
            if strengths[0] > 0.7:
                direction = directions[0]
                self.hop_selections['1-hop'] += 1
            elif strengths[1] > 0.5:
                direction = directions[1] * 0.7 + directions[0] * 0.3
                self.hop_selections['2-hop'] += 1
            elif strengths[2] > 0.4:
                direction = directions[2] * 0.6 + directions[1] * 0.3 + directions[0] * 0.1
                self.hop_selections['3-hop'] += 1
            else:
                # Weighted combination
                total = sum(strengths) + 0.001
                weights = [s / total for s in strengths]
                direction = sum(d * w for d, w in zip(directions[:5], weights))
                self.hop_selections['adaptive'] += 1
        
        # Enhanced action scoring
        action_scores = np.zeros(4)
        
        for i, action in enumerate(self.actions):
            dx, dy = self.action_deltas[action]
            
            # Position-based score
            action_vec = np.zeros(2)
            action_vec[0] = dx / self.height
            action_vec[1] = dy / self.width
            position_score = np.dot(direction[:2], action_vec)
            
            # Feature-based score
            feature_score = direction[2+i] * 0.5 if direction[2+i] > 0 else 0
            
            # Strong goal direction bonus
            goal_vec = np.array([self.goal[0] - x, self.goal[1] - y])
            goal_norm = np.linalg.norm(goal_vec)
            if goal_norm > 0:
                goal_vec = goal_vec / goal_norm
                goal_alignment = np.dot(goal_vec, [dx, dy])
            else:
                goal_alignment = 0
            
            # Combined score with strong goal bias
            action_scores[i] = max(0, 
                position_score * 0.4 + 
                feature_score * 0.2 + 
                goal_alignment * 0.4
            )
        
        # Add significant exploration noise when stuck
        if self.stuck_counter > 15:
            action_scores += np.random.random(4) * 0.3
        elif self.stuck_counter > 5:
            action_scores += np.random.random(4) * 0.1
        else:
            action_scores += np.random.random(4) * 0.02
        
        # Select action
        if np.sum(action_scores) > 0:
            probs = action_scores / np.sum(action_scores)
            return np.random.choice(self.actions, p=probs)
        
        # Fallback: biased random toward goal
        goal_dx = self.goal[0] - x
        goal_dy = self.goal[1] - y
        
        if abs(goal_dx) > abs(goal_dy):
            if goal_dx < 0:
                return np.random.choice(self.actions, p=[0.4, 0.2, 0.2, 0.2])  # Bias up
            else:
                return np.random.choice(self.actions, p=[0.2, 0.2, 0.4, 0.2])  # Bias down
        else:
            if goal_dy < 0:
                return np.random.choice(self.actions, p=[0.2, 0.2, 0.2, 0.4])  # Bias left
            else:
                return np.random.choice(self.actions, p=[0.2, 0.4, 0.2, 0.2])  # Bias right
    
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
                
                print(f"\n🎉🎉🎉 INCREDIBLE SUCCESS! Reached goal in {step} steps! 🎉🎉🎉")
                print(f"Pure episodic memory with {self.message_depth}-hop deep reasoning!")
                print(f"Cache performance: {self.cache_hits} hits, {self.cache_misses} misses")
                
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
                    'wall_hits': wall_hits,
                    'cache_hits': self.cache_hits,
                    'cache_misses': self.cache_misses
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
                      f"wall_hits={wall_hits} ({hit_rate:.1f}%), stuck={self.stuck_counter}")
                
                # Show hop usage periodically
                if step % 500 == 0:
                    active_hops = {k: v for k, v in self.hop_selections.items() if v > 0}
                    print(f"  Hop usage: {active_hops}")
                    print(f"  Cache: {self.cache_hits} hits, {self.cache_misses} misses")
        
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
            'wall_hits': wall_hits,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses
        }