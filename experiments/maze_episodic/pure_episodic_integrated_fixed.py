#!/usr/bin/env python3
"""
Fixed Pure Episodic Navigator with true O(1) search using integrated index
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add path for integrated index
sys.path.append(os.path.join(os.path.dirname(__file__), '../maze-optimized-search/src'))
from integrated_vector_graph_index import IntegratedVectorGraphIndex


class PureEpisodicIntegratedFixed:
    """Pure episodic navigator using integrated index with FAISS for O(1) search"""
    
    def __init__(self, maze: np.ndarray, message_depth: int = 3):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = self._find_start()
        self.goal = self._find_goal()
        self.message_depth = message_depth
        
        # Integrated index with FAISS enabled for true O(1) search
        self.index = IntegratedVectorGraphIndex(
            dimension=6,  # 2 position + 4 visual
            config={
                'similarity_threshold': 0.7,
                'use_faiss': True,  # Enable FAISS for O(1) search
                'faiss_threshold': 10,  # Use FAISS even for small datasets
                'max_edges_per_node': 10
            }
        )
        
        # Memory systems
        self.visual_memory = {}
        self.search_times = []
        self.path = [(self.position[0], self.position[1])]
        
        # Action mapping
        self.actions = ['up', 'right', 'down', 'left']
        self.action_deltas = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        
    def _find_start(self) -> Tuple[int, int]:
        """Find start position (usually top-left open space)"""
        for i in range(self.height):
            for j in range(self.width):
                if self.maze[i, j] == 0:
                    return (i, j)
        return (1, 1)
    
    def _find_goal(self) -> Tuple[int, int]:
        """Find goal position (usually bottom-right open space)"""
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
    
    def _create_episode_vector(self, x: int, y: int) -> np.ndarray:
        """Create 6D episode vector"""
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
    
    def _simple_message_passing(self, indices: List[int]) -> np.ndarray:
        """Simplified message passing that focuses on action selection"""
        if not indices:
            return np.zeros(4)  # 4 actions
            
        action_scores = np.zeros(4)
        
        # Aggregate successful actions from similar episodes
        for idx in indices:
            if idx < len(self.index.metadata):
                meta = self.index.metadata[idx]
                # Extract action from episode (if stored)
                if 'action' in meta:
                    action = meta['action']
                    if action in self.actions:
                        action_idx = self.actions.index(action)
                        action_scores[action_idx] += 1.0
                        
        # Normalize
        if np.sum(action_scores) > 0:
            action_scores = action_scores / np.sum(action_scores)
            
        return action_scores
    
    def get_action(self) -> Optional[str]:
        """Get next action using integrated index search"""
        self._update_visual_memory()
        
        x, y = self.position
        current_vec = self._create_episode_vector(x, y)
        
        # Search for similar episodes (true O(1) with FAISS)
        start_time = time.time()
        indices, scores = self.index.search(current_vec, k=20)
        search_time = (time.time() - start_time) * 1000
        self.search_times.append(search_time)
        
        # Get action preferences from similar episodes
        action_scores = self._simple_message_passing(indices.tolist())
        
        # Add exploration bonus for unvisited directions
        vis = self.visual_memory.get((x, y), {})
        for i, action in enumerate(self.actions):
            if vis.get(action) == 'path':
                # Check if we've been there recently
                dx, dy = self.action_deltas[action]
                next_pos = (x + dx, y + dy)
                if next_pos not in [p for p in self.path[-10:]]:  # Not in recent path
                    action_scores[i] += 0.3  # Exploration bonus
        
        # Goal bias if we have goal information
        goal_vec = self._create_episode_vector(*self.goal)
        goal_similarity = np.dot(current_vec[:2], goal_vec[:2])  # Position similarity
        if goal_similarity > 0.8:
            # We're close to goal, bias towards it
            gx, gy = self.goal
            if x < gx and vis.get('down') == 'path':
                action_scores[self.actions.index('down')] += 0.5
            if x > gx and vis.get('up') == 'path':
                action_scores[self.actions.index('up')] += 0.5
            if y < gy and vis.get('right') == 'path':
                action_scores[self.actions.index('right')] += 0.5
            if y > gy and vis.get('left') == 'path':
                action_scores[self.actions.index('left')] += 0.5
        
        # Select action based on scores
        valid_actions = []
        for i, action in enumerate(self.actions):
            if vis.get(action) == 'path' and action_scores[i] > 0:
                valid_actions.append((action, action_scores[i]))
        
        if not valid_actions:
            # No good options, try any open path
            for action in self.actions:
                if vis.get(action) == 'path':
                    return action
            return None
        
        # Probabilistic selection
        actions, scores = zip(*valid_actions)
        scores = np.array(scores)
        if np.sum(scores) > 0:
            probs = scores / np.sum(scores)
            return np.random.choice(actions, p=probs)
        else:
            return np.random.choice(actions)
    
    def move(self, action: str) -> bool:
        """Execute action and store episode"""
        if action not in self.actions:
            return False
            
        dx, dy = self.action_deltas[action]
        new_x, new_y = self.position[0] + dx, self.position[1] + dy
        
        if (0 <= new_x < self.height and 0 <= new_y < self.width and 
            self.maze[new_x, new_y] == 0):
            
            # Store episode with action information
            episode_vec = self._create_episode_vector(*self.position)
            self.index.add_episode({
                'vec': episode_vec,
                'text': f"From ({self.position[0]}, {self.position[1]}) action {action}",
                'pos': self.position,
                'action': action,  # Store the successful action
                'c_value': 0.7  # Higher confidence for successful moves
            })
            
            # Update position
            self.position = (new_x, new_y)
            self.path.append(self.position)
            return True
        
        return False
    
    def navigate(self, max_steps: int = 5000) -> Dict:
        """Navigate maze and return results"""
        start_time = time.time()
        
        for step in range(max_steps):
            if self.position == self.goal:
                total_time = time.time() - start_time
                return {
                    'success': True,
                    'steps': step,
                    'total_episodes': len(self.index.metadata),
                    'total_time': total_time,
                    'avg_search_time': np.mean(self.search_times) if self.search_times else 0,
                    'total_search_time': np.sum(self.search_times) if self.search_times else 0,
                    'search_times': self.search_times,
                    'path': self.path
                }
            
            action = self.get_action()
            if action:
                self.move(action)
                
            if step % 100 == 0 and step > 0:
                avg_search = np.mean(self.search_times[-10:]) if len(self.search_times) >= 10 else np.mean(self.search_times)
                print(f"Step {step}: pos={self.position}, episodes={len(self.index.metadata)}, "
                      f"avg_search={avg_search:.2f}ms")
        
        total_time = time.time() - start_time
        return {
            'success': False,
            'steps': max_steps,
            'total_episodes': len(self.index.metadata),
            'total_time': total_time,
            'avg_search_time': np.mean(self.search_times) if self.search_times else 0,
            'total_search_time': np.sum(self.search_times) if self.search_times else 0,
            'search_times': self.search_times,
            'path': self.path
        }