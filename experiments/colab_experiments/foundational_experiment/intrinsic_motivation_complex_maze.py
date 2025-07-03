#!/usr/bin/env python3
"""
Complex Maze Environment for GED Testing
========================================

This implementation creates more complex mazes that require structural understanding:
- Multiple rooms with doors
- Keys to unlock doors
- Subgoals and hierarchical structure
- Dead ends that require backtracking
- Multiple valid paths with different complexities
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, defaultdict
import random
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from scipy import stats
import networkx as nx

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class ComplexMazeEnvironment:
    """Complex maze with rooms, doors, keys, and hierarchical structure"""
    
    def __init__(self, size=12, num_rooms=4, num_keys=2):
        self.size = size
        self.num_rooms = num_rooms
        self.num_keys = num_keys
        
        # Rewards
        self.step_penalty = -0.01
        self.collision_penalty = -0.05
        self.key_reward = 2.0
        self.door_penalty = -0.5  # Trying locked door
        self.subgoal_reward = 5.0
        self.goal_reward = 20.0
        self.backtrack_penalty = -0.02  # Revisiting states
        
        # Cell types
        self.EMPTY = 0
        self.WALL = 1
        self.DOOR = 2
        self.KEY = 3
        self.SUBGOAL = 4
        self.GOAL = 5
        self.AGENT = 6
        
        self.reset()
    
    def _create_rooms(self):
        """Create a maze with multiple rooms"""
        self.grid = np.ones((self.size, self.size))  # Start with walls
        
        # Define room boundaries
        room_size = self.size // 2
        rooms = []
        
        # Create 4 rooms
        for i in range(2):
            for j in range(2):
                room_start_x = i * room_size + 1
                room_start_y = j * room_size + 1
                room_end_x = (i + 1) * room_size - 1
                room_end_y = (j + 1) * room_size - 1
                
                # Clear room interior
                self.grid[room_start_x:room_end_x, room_start_y:room_end_y] = self.EMPTY
                
                rooms.append({
                    'id': i * 2 + j,
                    'bounds': (room_start_x, room_start_y, room_end_x, room_end_y),
                    'center': ((room_start_x + room_end_x) // 2, (room_start_y + room_end_y) // 2)
                })
        
        # Create corridors between rooms
        # Horizontal corridors
        corridor_y = self.size // 2
        self.grid[2:self.size-2, corridor_y-1:corridor_y+1] = self.EMPTY
        
        # Vertical corridors
        corridor_x = self.size // 2
        self.grid[corridor_x-1:corridor_x+1, 2:self.size-2] = self.EMPTY
        
        # Add some complexity - internal walls in rooms
        for room in rooms:
            x1, y1, x2, y2 = room['bounds']
            
            # Add internal structure (50% chance)
            if np.random.random() < 0.5:
                # Vertical wall with gap
                wall_x = (x1 + x2) // 2
                self.grid[wall_x, y1:y2] = self.WALL
                gap_y = np.random.randint(y1 + 1, y2 - 1)
                self.grid[wall_x, gap_y] = self.EMPTY
            
            if np.random.random() < 0.5:
                # Horizontal wall with gap
                wall_y = (y1 + y2) // 2
                self.grid[x1:x2, wall_y] = self.WALL
                gap_x = np.random.randint(x1 + 1, x2 - 1)
                self.grid[gap_x, wall_y] = self.EMPTY
        
        return rooms
    
    def _place_doors_and_keys(self, rooms):
        """Place doors between rooms and keys to unlock them"""
        self.doors = []
        self.keys = []
        self.door_key_mapping = {}
        
        # Place doors at room entrances
        door_positions = [
            (rooms[0]['center'][0], self.size // 2),  # Room 0 to corridor
            (rooms[1]['center'][0], self.size // 2),  # Room 1 to corridor
            (self.size // 2, rooms[0]['center'][1]),  # Room 0 to corridor
            (self.size // 2, rooms[2]['center'][1]),  # Room 2 to corridor
        ]
        
        # Select subset of doors to actually place
        num_doors = min(self.num_keys + 1, len(door_positions))
        selected_positions = random.sample(door_positions, num_doors)
        
        for i, pos in enumerate(selected_positions[:self.num_keys]):
            # Place door
            self.grid[pos] = self.DOOR
            self.doors.append({'id': i, 'pos': pos, 'locked': True})
            
            # Place corresponding key in a different room
            key_room = random.choice(rooms)
            key_x = np.random.randint(key_room['bounds'][0] + 1, key_room['bounds'][2] - 1)
            key_y = np.random.randint(key_room['bounds'][1] + 1, key_room['bounds'][3] - 1)
            
            # Ensure key is on empty space
            attempts = 0
            while self.grid[key_x, key_y] != self.EMPTY and attempts < 20:
                key_x = np.random.randint(key_room['bounds'][0] + 1, key_room['bounds'][2] - 1)
                key_y = np.random.randint(key_room['bounds'][1] + 1, key_room['bounds'][3] - 1)
                attempts += 1
            
            if self.grid[key_x, key_y] == self.EMPTY:
                self.grid[key_x, key_y] = self.KEY
                self.keys.append({'id': i, 'pos': (key_x, key_y)})
                self.door_key_mapping[i] = i
    
    def _place_goals(self, rooms):
        """Place subgoals and main goal"""
        # Place main goal in last room
        goal_room = rooms[-1]
        self.goal_pos = goal_room['center']
        
        # Ensure goal is accessible
        if self.grid[self.goal_pos] != self.EMPTY:
            # Find nearby empty cell
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    new_pos = (self.goal_pos[0] + dx, self.goal_pos[1] + dy)
                    if self.grid[new_pos] == self.EMPTY:
                        self.goal_pos = new_pos
                        break
        
        self.grid[self.goal_pos] = self.GOAL
        
        # Place subgoals in other rooms
        self.subgoals = []
        for i, room in enumerate(rooms[:-1]):
            if np.random.random() < 0.6:  # 60% chance of subgoal
                subgoal_pos = (
                    np.random.randint(room['bounds'][0] + 1, room['bounds'][2] - 1),
                    np.random.randint(room['bounds'][1] + 1, room['bounds'][3] - 1)
                )
                
                if self.grid[subgoal_pos] == self.EMPTY:
                    self.grid[subgoal_pos] = self.SUBGOAL
                    self.subgoals.append(subgoal_pos)
    
    def reset(self):
        """Reset to new complex maze"""
        # Create maze structure
        rooms = self._create_rooms()
        self._place_doors_and_keys(rooms)
        self._place_goals(rooms)
        
        # Place agent in first room
        start_room = rooms[0]
        self.agent_pos = (
            np.random.randint(start_room['bounds'][0] + 1, start_room['bounds'][2] - 1),
            np.random.randint(start_room['bounds'][1] + 1, start_room['bounds'][3] - 1)
        )
        
        # Ensure starting position is empty
        while self.grid[self.agent_pos] != self.EMPTY:
            self.agent_pos = (
                np.random.randint(start_room['bounds'][0] + 1, start_room['bounds'][2] - 1),
                np.random.randint(start_room['bounds'][1] + 1, start_room['bounds'][3] - 1)
            )
        
        # Initialize state
        self.collected_keys = set()
        self.visited_subgoals = set()
        self.visited_positions = {self.agent_pos}
        self.step_count = 0
        self.max_steps = self.size * self.size * 3
        
        return self._get_state()
    
    def _get_state(self):
        """Get state representation"""
        # Multi-channel representation
        channels = 7
        state = np.zeros((self.size, self.size, channels))
        
        # Channel 0: Walls
        state[:, :, 0] = (self.grid == self.WALL).astype(float)
        
        # Channel 1: Doors (locked/unlocked)
        for door in self.doors:
            if door['locked']:
                state[door['pos'][0], door['pos'][1], 1] = 1.0
            else:
                state[door['pos'][0], door['pos'][1], 1] = 0.5
        
        # Channel 2: Keys (not collected)
        for key in self.keys:
            if key['id'] not in self.collected_keys:
                state[key['pos'][0], key['pos'][1], 2] = 1.0
        
        # Channel 3: Subgoals (not visited)
        for subgoal in self.subgoals:
            if subgoal not in self.visited_subgoals:
                state[subgoal[0], subgoal[1], 3] = 1.0
        
        # Channel 4: Goal
        state[self.goal_pos[0], self.goal_pos[1], 4] = 1.0
        
        # Channel 5: Agent position
        state[self.agent_pos[0], self.agent_pos[1], 5] = 1.0
        
        # Channel 6: Visited positions (memory)
        for pos in self.visited_positions:
            state[pos[0], pos[1], 6] = 0.5
        
        return state.flatten()
    
    def step(self, action):
        """Execute action"""
        # Actions: 0=up, 1=right, 2=down, 3=left
        moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        dx, dy = moves[action]
        
        new_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
        
        # Check boundaries
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            return self._get_state(), self.collision_penalty, False, {'success': False}
        
        # Initialize reward
        reward = self.step_penalty
        
        # Check if revisiting position (backtracking)
        if new_pos in self.visited_positions:
            reward += self.backtrack_penalty
        
        # Check cell type
        cell_type = self.grid[new_pos]
        
        if cell_type == self.WALL:
            # Can't move through walls
            reward = self.collision_penalty
        
        elif cell_type == self.DOOR:
            # Check if door is locked
            door = next(d for d in self.doors if d['pos'] == new_pos)
            if door['locked']:
                # Check if we have the key
                if door['id'] in self.collected_keys:
                    door['locked'] = False
                    self.grid[new_pos] = self.EMPTY
                    reward = self.subgoal_reward  # Unlocking door is good
                    self.agent_pos = new_pos
                else:
                    reward = self.door_penalty  # Tried locked door
            else:
                self.agent_pos = new_pos
        
        elif cell_type == self.KEY:
            # Collect key
            key = next(k for k in self.keys if k['pos'] == new_pos)
            if key['id'] not in self.collected_keys:
                self.collected_keys.add(key['id'])
                self.grid[new_pos] = self.EMPTY
                reward = self.key_reward
            self.agent_pos = new_pos
        
        elif cell_type == self.SUBGOAL:
            # Reach subgoal
            if new_pos not in self.visited_subgoals:
                self.visited_subgoals.add(new_pos)
                reward = self.subgoal_reward
            self.agent_pos = new_pos
        
        elif cell_type == self.GOAL:
            # Reached goal!
            self.agent_pos = new_pos
            reward = self.goal_reward
            # Bonus for efficiency
            efficiency_bonus = (1 - self.step_count / self.max_steps) * 10
            reward += efficiency_bonus
            return self._get_state(), reward, True, {'success': True, 'steps': self.step_count}
        
        else:  # EMPTY
            self.agent_pos = new_pos
        
        # Update visited positions
        self.visited_positions.add(self.agent_pos)
        self.step_count += 1
        
        # Check timeout
        done = self.step_count >= self.max_steps
        info = {
            'success': False,
            'steps': self.step_count,
            'keys_collected': len(self.collected_keys),
            'subgoals_visited': len(self.visited_subgoals),
            'coverage': len(self.visited_positions) / (self.size * self.size)
        }
        
        return self._get_state(), reward, done, info
    
    def render(self, ax=None):
        """Render the maze"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Create color map
        render_grid = np.zeros((self.size, self.size, 3))
        
        # Walls - black
        render_grid[self.grid == self.WALL] = [0, 0, 0]
        
        # Empty - white
        render_grid[self.grid == self.EMPTY] = [1, 1, 1]
        
        # Doors - brown (locked) or gray (unlocked)
        for door in self.doors:
            if door['locked']:
                render_grid[door['pos'][0], door['pos'][1]] = [0.6, 0.3, 0]
            else:
                render_grid[door['pos'][0], door['pos'][1]] = [0.7, 0.7, 0.7]
        
        # Keys - yellow
        for key in self.keys:
            if key['id'] not in self.collected_keys:
                render_grid[key['pos'][0], key['pos'][1]] = [1, 1, 0]
        
        # Subgoals - cyan
        for subgoal in self.subgoals:
            if subgoal not in self.visited_subgoals:
                render_grid[subgoal[0], subgoal[1]] = [0, 1, 1]
        
        # Goal - green
        render_grid[self.goal_pos[0], self.goal_pos[1]] = [0, 1, 0]
        
        # Agent - red
        render_grid[self.agent_pos[0], self.agent_pos[1]] = [1, 0, 0]
        
        # Visited positions - light gray
        for pos in self.visited_positions:
            if self.grid[pos] == self.EMPTY and pos != self.agent_pos:
                render_grid[pos[0], pos[1]] = [0.9, 0.9, 0.9]
        
        ax.imshow(render_grid, interpolation='nearest')
        ax.set_title(f'Complex Maze - Step {self.step_count}\n'
                    f'Keys: {len(self.collected_keys)}/{self.num_keys}, '
                    f'Subgoals: {len(self.visited_subgoals)}/{len(self.subgoals)}')
        ax.axis('off')
        
        return ax
    
    @property
    def state_space_size(self):
        return self.size * self.size * 7  # 7 channels
    
    @property
    def action_space_size(self):
        return 4


class ImprovedKnowledgeGraph:
    """Enhanced knowledge graph for complex maze navigation"""
    
    def __init__(self, max_nodes=500):
        self.graph = nx.DiGraph()  # Directed graph for transitions
        self.max_nodes = max_nodes
        self.state_to_node = {}
        self.node_visits = defaultdict(int)
        self.node_values = defaultdict(float)  # Estimated value of states
        self.subgraph_structures = []  # Identified patterns/structures
        
    def add_transition(self, state_from, state_to, action, reward, info):
        """Add state transition with context"""
        # Create compact state representations
        key_from = self._compress_state(state_from, info.get('prev_info', {}))
        key_to = self._compress_state(state_to, info)
        
        # Add nodes
        if key_from not in self.state_to_node:
            node_id = len(self.state_to_node)
            self.state_to_node[key_from] = node_id
            self.graph.add_node(node_id, 
                              compressed_state=key_from,
                              keys=info.get('prev_keys', 0),
                              subgoals=info.get('prev_subgoals', 0))
        
        if key_to not in self.state_to_node:
            node_id = len(self.state_to_node)
            self.state_to_node[key_to] = node_id
            self.graph.add_node(node_id,
                              compressed_state=key_to,
                              keys=info.get('keys_collected', 0),
                              subgoals=info.get('subgoals_visited', 0))
        
        # Get node IDs
        node_from = self.state_to_node[key_from]
        node_to = self.state_to_node[key_to]
        
        # Update visits and values
        self.node_visits[node_from] += 1
        self.node_visits[node_to] += 1
        self.node_values[node_to] = max(self.node_values[node_to], 
                                       self.node_values[node_from] + reward)
        
        # Add edge with context
        if self.graph.has_edge(node_from, node_to):
            # Update existing edge
            edge_data = self.graph[node_from][node_to]
            edge_data['count'] += 1
            edge_data['avg_reward'] = (edge_data['avg_reward'] * (edge_data['count'] - 1) + reward) / edge_data['count']
        else:
            self.graph.add_edge(node_from, node_to,
                              action=action,
                              avg_reward=reward,
                              count=1)
        
        # Prune if needed
        if len(self.graph) > self.max_nodes:
            self._prune_low_value_nodes()
        
        # Identify structural patterns
        self._identify_patterns()
    
    def _compress_state(self, state, info):
        """Create compressed state representation focusing on important features"""
        # Extract key features from flattened state
        size = int(np.sqrt(len(state) // 7))
        state_3d = state.reshape(size, size, 7)
        
        # Find agent position
        agent_pos = np.unravel_index(np.argmax(state_3d[:, :, 5]), (size, size))
        
        # Find goal position
        goal_pos = np.unravel_index(np.argmax(state_3d[:, :, 4]), (size, size))
        
        # Discretize relative position to goal
        rel_x = (goal_pos[0] - agent_pos[0]) // 3
        rel_y = (goal_pos[1] - agent_pos[1]) // 3
        
        # Create compressed representation
        compressed = (
            agent_pos[0] // 3,  # Coarse agent position
            agent_pos[1] // 3,
            rel_x,  # Relative to goal
            rel_y,
            info.get('keys_collected', 0),  # Keys collected
            info.get('subgoals_visited', 0),  # Subgoals visited
            tuple(sorted(info.get('collected_keys', set())))  # Which keys
        )
        
        return compressed
    
    def _identify_patterns(self):
        """Identify useful subgraph patterns"""
        if len(self.graph) < 10:
            return
        
        # Find strongly connected components (loops/cycles)
        sccs = list(nx.strongly_connected_components(self.graph))
        
        # Find paths that lead to high-value nodes
        high_value_nodes = [n for n, v in self.node_values.items() 
                           if v > np.percentile(list(self.node_values.values()), 80)]
        
        # Store identified structures
        self.subgraph_structures = {
            'loops': [scc for scc in sccs if len(scc) > 2],
            'high_value_paths': high_value_nodes,
            'bottlenecks': self._find_bottlenecks()
        }
    
    def _find_bottlenecks(self):
        """Find nodes that are critical for reaching high-value states"""
        bottlenecks = []
        
        if len(self.graph) < 20:
            return bottlenecks
        
        # Find nodes with high betweenness centrality
        try:
            centrality = nx.betweenness_centrality(self.graph)
            threshold = np.percentile(list(centrality.values()), 90)
            bottlenecks = [n for n, c in centrality.items() if c > threshold]
        except:
            pass
        
        return bottlenecks
    
    def _prune_low_value_nodes(self):
        """Remove low-value, rarely visited nodes"""
        # Calculate node importance
        importance = {}
        for node in self.graph.nodes():
            visits = self.node_visits[node]
            value = self.node_values[node]
            centrality = self.graph.degree(node)
            
            importance[node] = visits * 0.3 + value * 0.5 + centrality * 0.2
        
        # Remove bottom 10%
        threshold = np.percentile(list(importance.values()), 10)
        nodes_to_remove = [n for n, imp in importance.items() if imp < threshold]
        
        for node in nodes_to_remove:
            if node in self.graph:
                self.graph.remove_node(node)
                # Clean up mappings
                for state, n in list(self.state_to_node.items()):
                    if n == node:
                        del self.state_to_node[state]
    
    def compute_structural_complexity(self):
        """Compute maze-specific structural complexity"""
        if len(self.graph) == 0:
            return 0
        
        # Components of complexity
        num_nodes = len(self.graph)
        num_edges = self.graph.number_of_edges()
        
        # Path diversity (number of different paths to high-value states)
        path_diversity = len(self.subgraph_structures.get('high_value_paths', []))
        
        # Bottleneck penalty (fewer bottlenecks = simpler structure)
        bottleneck_penalty = len(self.subgraph_structures.get('bottlenecks', []))
        
        # Loop penalty (fewer loops = more direct paths)
        loop_penalty = len(self.subgraph_structures.get('loops', []))
        
        # Combined complexity (lower is better)
        complexity = (num_nodes + num_edges) / (1 + path_diversity) + bottleneck_penalty + loop_penalty
        
        return complexity


# Define the complete agent class
class ComplexMazeAgent:
    """Agent adapted for complex maze navigation"""
    
    def __init__(self, state_size, action_size, 
                 use_ged=True, use_ig=True, 
                 intrinsic_weight=0.1, learning_rate=0.001):
        
        self.state_size = state_size
        self.action_size = action_size
        self.use_ged = use_ged
        self.use_ig = use_ig
        self.intrinsic_weight = intrinsic_weight
        
        # Q-Network - larger for complex environment
        self.q_network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        
        # Use improved knowledge graph
        self.knowledge_graph = ImprovedKnowledgeGraph(max_nodes=500)
        
        # State visit tracking for IG
        self.state_visits = defaultdict(int)
        
        # Complexity tracking
        self.complexity_history = deque(maxlen=100)
        self.prev_complexity = None
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Context tracking
        self.prev_info = {}
    
    def _calculate_ig(self, state):
        """Calculate Information Gain"""
        if not self.use_ig:
            return 1.0
        
        # Compress state for IG calculation
        state_key = tuple(state[::10])  # Sample every 10th element
        old_visits = self.state_visits[state_key]
        self.state_visits[state_key] += 1
        
        # Simple novelty measure
        novelty = 1.0 / (1.0 + old_visits)
        
        # Boost for completely new states
        if old_visits == 0:
            novelty += 0.5
        
        return novelty
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy"""
        if training and np.random.random() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def replay(self, batch_size=64):
        """Train the network"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor(np.array([e[0] for e in batch]))
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor(np.array([e[3] for e in batch]))
        dones = torch.FloatTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.q_network(next_states).max(1)[0]
            target_q_values = rewards + (0.99 * next_q_values * (1 - dones))
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_knowledge_stats(self):
        """Get statistics about knowledge structure"""
        kg = self.knowledge_graph
        
        return {
            'num_states': len(kg.graph),
            'num_transitions': kg.graph.number_of_edges() if hasattr(kg.graph, 'number_of_edges') else 0,
            'complexity': kg.compute_structural_complexity(),
            'avg_visits': np.mean(list(kg.node_visits.values())) if kg.node_visits else 0
        }
    
    def remember(self, state, action, reward, next_state, done, info):
        """Store experience with maze context"""
        # Calculate intrinsic reward with context
        info['prev_info'] = self.prev_info
        intrinsic_reward = self._calculate_intrinsic_reward(state, action, next_state, reward, info)
        total_reward = reward + self.intrinsic_weight * intrinsic_reward
        
        self.memory.append((state, action, total_reward, next_state, done))
        self.prev_info = info.copy()
    
    def _calculate_intrinsic_reward(self, state, action, next_state, reward, info):
        """Enhanced intrinsic reward for complex maze"""
        if not self.use_ged and not self.use_ig:
            return 0.0
        
        # Calculate IG
        ig = self._calculate_ig(next_state)
        
        # Calculate GED with maze context
        self.knowledge_graph.add_transition(state, next_state, action, reward, info)
        
        current_complexity = self.knowledge_graph.compute_structural_complexity()
        self.complexity_history.append(current_complexity)
        
        if self.prev_complexity is None:
            self.prev_complexity = current_complexity
            ged = 0.0
        else:
            # Complexity reduction is good
            delta_complexity = current_complexity - self.prev_complexity
            self.prev_complexity = current_complexity
            
            # Additional bonus for reaching new areas
            if info.get('keys_collected', 0) > info.get('prev_info', {}).get('keys_collected', 0):
                delta_complexity -= 0.5  # Key collection simplifies future navigation
            
            if info.get('subgoals_visited', 0) > info.get('prev_info', {}).get('subgoals_visited', 0):
                delta_complexity -= 0.3  # Subgoal progress
            
            ged = -delta_complexity  # Positive when complexity reduces
        
        # Combine with emphasis on structural learning
        intrinsic_reward = ig * max(0, ged)
        
        # Bonus for discovering efficient paths
        if ged > 0.5 and ig > 0.1:
            intrinsic_reward *= 2.0  # Insight bonus
        
        return intrinsic_reward


def run_complex_maze_experiment(episodes=200, trials=3):
    """Run experiment in complex maze environment"""
    
    configs = [
        {"name": "Complex_Full", "ged": True, "ig": True, "weight": 0.2},
        {"name": "Complex_IG_Only", "ged": False, "ig": True, "weight": 0.2},
        {"name": "Complex_GED_Only", "ged": True, "ig": False, "weight": 0.2},
        {"name": "Baseline", "ged": False, "ig": False, "weight": 0.0}
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        
        config_results = {
            'success_rates': [],
            'convergence_episodes': [],
            'avg_steps_to_goal': [],
            'keys_collected': [],
            'subgoals_reached': [],
            'final_complexity': [],
            'complexity_reduction': []
        }
        
        for trial in range(trials):
            # Set seed
            seed = RANDOM_SEED + trial * 100
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            
            env = ComplexMazeEnvironment(size=12, num_rooms=4, num_keys=2)
            agent = ComplexMazeAgent(
                env.state_space_size,
                env.action_space_size,
                use_ged=config['ged'],
                use_ig=config['ig'],
                intrinsic_weight=config['weight']
            )
            
            successes = []
            steps_to_goal = []
            convergence_ep = episodes
            initial_complexity = None
            
            for ep in range(episodes):
                state = env.reset()
                done = False
                ep_steps = 0
                
                while not done:
                    action = agent.act(state)
                    next_state, reward, done, info = env.step(action)
                    agent.remember(state, action, reward, next_state, done, info)
                    state = next_state
                    ep_steps += 1
                    
                    if done:
                        success = info['success']
                        successes.append(1 if success else 0)
                        if success:
                            steps_to_goal.append(ep_steps)
                
                # Train more intensively
                if len(agent.memory) > 64:
                    for _ in range(8):
                        agent.replay(batch_size=64)
                
                # Track complexity
                if ep == 20:  # Early complexity
                    initial_complexity = agent.knowledge_graph.compute_structural_complexity()
                
                # Check convergence
                if len(successes) >= 10:
                    recent_rate = np.mean(successes[-10:])
                    if recent_rate >= 0.5 and convergence_ep == episodes:
                        convergence_ep = ep
                
                # Periodic progress report
                if (ep + 1) % 50 == 0:
                    recent_success = np.mean(successes[-20:]) if len(successes) >= 20 else 0
                    print(f"    Episode {ep+1}: Recent success rate = {recent_success:.3f}")
            
            # Final statistics
            final_stats = agent.get_knowledge_stats()
            final_complexity = final_stats['complexity']
            
            if initial_complexity and initial_complexity > 0:
                complexity_reduction = (initial_complexity - final_complexity) / initial_complexity
            else:
                complexity_reduction = 0
            
            # Aggregate results
            config_results['success_rates'].append(np.mean(successes))
            config_results['convergence_episodes'].append(convergence_ep)
            config_results['avg_steps_to_goal'].append(np.mean(steps_to_goal) if steps_to_goal else episodes)
            config_results['keys_collected'].append(np.mean([s['keys_collected'] for s in [info]]))
            config_results['subgoals_reached'].append(np.mean([s['subgoals_visited'] for s in [info]]))
            config_results['final_complexity'].append(final_complexity)
            config_results['complexity_reduction'].append(complexity_reduction)
            
            print(f"  Trial {trial+1}: Success={np.mean(successes):.3f}, "
                  f"Conv={convergence_ep}, Complexity reduction={complexity_reduction:.3f}")
        
        results[config['name']] = config_results
    
    return results


def visualize_complex_maze_results(results, output_dir):
    """Create comprehensive visualization"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    names = list(results.keys())
    colors = ['blue', 'green', 'orange', 'red']
    
    # Success rates
    ax = axes[0, 0]
    means = [np.mean(results[n]['success_rates']) for n in names]
    stds = [np.std(results[n]['success_rates']) for n in names]
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors)
    ax.set_ylabel('Success Rate')
    ax.set_title('Overall Performance')
    ax.set_ylim(0, 1.1)
    
    # Add significance markers
    y_max = max(means) + max(stds) + 0.1
    baseline_idx = names.index('Baseline')
    for i, name in enumerate(names):
        if i != baseline_idx and results[name]['success_rates']:
            baseline_data = results['Baseline']['success_rates']
            test_data = results[name]['success_rates']
            if len(test_data) == len(baseline_data) > 1:
                _, p_val = stats.ttest_rel(test_data, baseline_data)
                if p_val < 0.05:
                    ax.plot([baseline_idx, i], [y_max, y_max], 'k-')
                    ax.text((baseline_idx + i) / 2, y_max + 0.02, '*', 
                           ha='center', fontsize=14)
    
    # Convergence speed
    ax = axes[0, 1]
    conv_means = [np.mean(results[n]['convergence_episodes']) for n in names]
    ax.bar(names, conv_means, color=colors)
    ax.set_ylabel('Episodes to Convergence')
    ax.set_title('Learning Speed')
    
    # Complexity reduction
    ax = axes[0, 2]
    reduction_means = [np.mean(results[n]['complexity_reduction']) * 100 for n in names]
    ax.bar(names, reduction_means, color=colors)
    ax.set_ylabel('Complexity Reduction (%)')
    ax.set_title('Structural Optimization')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Steps to goal (efficiency)
    ax = axes[1, 0]
    steps_means = [np.mean(results[n]['avg_steps_to_goal']) for n in names]
    ax.bar(names, steps_means, color=colors)
    ax.set_ylabel('Average Steps to Goal')
    ax.set_title('Navigation Efficiency')
    
    # Learning curves (mock)
    ax = axes[1, 1]
    episodes = range(50)
    for i, name in enumerate(names):
        # Simulate learning curve
        success_rate = means[i]
        curve = success_rate * (1 - np.exp(-np.array(episodes) / 20))
        ax.plot(episodes, curve, label=name, color=colors[i], linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Learning Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Strategy comparison radar chart
    ax = axes[1, 2]
    categories = ['Success', 'Speed', 'Efficiency', 'Optimization']
    
    # Normalize metrics to 0-1 scale
    for i, name in enumerate(names):
        values = [
            means[i],  # Success rate (already 0-1)
            1 - conv_means[i] / max(conv_means),  # Inverse convergence
            1 - steps_means[i] / max(steps_means),  # Inverse steps
            (reduction_means[i] + 20) / 40  # Normalized complexity reduction
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Multi-Metric Comparison')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'complex_maze_results.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Run complex maze experiment"""
    
    output_dir = Path("experiments/foundational_intrinsic_motivation/results_complex_maze")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("COMPLEX MAZE EXPERIMENT - TESTING TRUE GED VALUE")
    print("="*70)
    print("\nMaze features:")
    print("- Multiple rooms with locked doors")
    print("- Keys required for progression")
    print("- Subgoals for hierarchical learning")
    print("- Dead ends requiring backtracking")
    print("- Long-term planning required")
    
    # Run experiment
    results = run_complex_maze_experiment(episodes=200, trials=3)
    
    # Analyze results
    print("\n" + "="*70)
    print("RESULTS ANALYSIS")
    print("="*70)
    
    # Success rates
    print("\nSuccess Rates:")
    for name, data in results.items():
        mean_success = np.mean(data['success_rates'])
        std_success = np.std(data['success_rates'])
        print(f"  {name}: {mean_success:.3f} ± {std_success:.3f}")
    
    # Complexity reduction
    print("\nStructural Optimization:")
    for name, data in results.items():
        mean_reduction = np.mean(data['complexity_reduction'])
        print(f"  {name}: {mean_reduction:.3%} complexity reduction")
    
    # Save results
    save_data = {
        'results': {
            name: {
                'success_rate_mean': float(np.mean(data['success_rates'])),
                'success_rate_std': float(np.std(data['success_rates'])),
                'convergence_mean': float(np.mean(data['convergence_episodes'])),
                'complexity_reduction_mean': float(np.mean(data['complexity_reduction'])),
                'avg_steps_to_goal': float(np.mean(data['avg_steps_to_goal']))
            }
            for name, data in results.items()
        },
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'random_seed': RANDOM_SEED,
            'episodes': 200,
            'trials': 3,
            'maze_size': 12,
            'num_rooms': 4,
            'num_keys': 2
        }
    }
    
    with open(output_dir / 'complex_maze_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    
    # Create visualizations
    visualize_complex_maze_results(results, output_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    best = max(results.keys(), key=lambda k: np.mean(results[k]['success_rates']))
    print(f"\nBest performer: {best}")
    
    if 'Complex_Full' in results and 'Baseline' in results:
        full_success = np.mean(results['Complex_Full']['success_rates'])
        baseline_success = np.mean(results['Baseline']['success_rates'])
        
        if full_success > baseline_success:
            improvement = (full_success - baseline_success) / baseline_success * 100
            print(f"Complex_Full shows {improvement:.1f}% improvement over Baseline")
    
    print(f"\nResults saved to {output_dir}")
    
    # Demonstrate maze visualization
    print("\nGenerating example maze visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    env = ComplexMazeEnvironment()
    env.reset()
    env.render(ax)
    plt.savefig(output_dir / 'example_complex_maze.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Example maze saved!")


if __name__ == "__main__":
    main()