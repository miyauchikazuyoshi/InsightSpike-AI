#!/usr/bin/env python3
"""
Simple Message Passing Navigator
================================

Simplified version with direct message passing effect demonstration.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import time

@dataclass
class EpisodeNode:
    """Episode as a node in the graph"""
    id: int
    pos: Tuple[int, int]
    action: int
    reward: float
    embedding: np.ndarray

class SimpleMessagePassing:
    """Simple message passing for maze navigation"""
    
    def __init__(self, maze: np.ndarray, num_hops: int = 2):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.visited = {(1, 1)}
        self.num_hops = num_hops
        
        # Episode nodes
        self.nodes: List[EpisodeNode] = []
        self.adjacency = {}  # node_id -> list of neighbor ids
        
        # Statistics
        self.wall_hits = 0
        self.moves = 0
        
    def add_episode(self, pos: Tuple[int, int], action: int, reward: float):
        """Add episode and connect to nearby episodes"""
        # Create embedding (simple: position + action + reward)
        embedding = np.array([
            pos[0] / self.width,
            pos[1] / self.height,
            action / 3.0,
            reward
        ])
        
        node = EpisodeNode(
            id=len(self.nodes),
            pos=pos,
            action=action,
            reward=reward,
            embedding=embedding
        )
        
        # Connect to spatially close episodes
        self.adjacency[node.id] = []
        for other in self.nodes:
            dist = abs(pos[0] - other.pos[0]) + abs(pos[1] - other.pos[1])
            if dist <= 2:  # Within 2 Manhattan distance
                self.adjacency[node.id].append(other.id)
                if other.id in self.adjacency:
                    self.adjacency[other.id].append(node.id)
        
        self.nodes.append(node)
    
    def message_pass(self):
        """Simple message passing: average neighbor embeddings"""
        if len(self.nodes) < 2:
            return
        
        # Multiple rounds of message passing
        for _ in range(self.num_hops):
            new_embeddings = []
            
            for node in self.nodes:
                if node.id in self.adjacency and self.adjacency[node.id]:
                    # Average neighbor embeddings
                    neighbor_embeddings = []
                    for neighbor_id in self.adjacency[node.id]:
                        neighbor_embeddings.append(self.nodes[neighbor_id].embedding)
                    
                    # Mix with own embedding
                    avg_neighbors = np.mean(neighbor_embeddings, axis=0)
                    new_embedding = 0.7 * node.embedding + 0.3 * avg_neighbors
                else:
                    new_embedding = node.embedding
                
                new_embeddings.append(new_embedding)
            
            # Update embeddings
            for i, node in enumerate(self.nodes):
                node.embedding = new_embeddings[i]
    
    def get_action_value(self, pos: Tuple[int, int], action: int) -> float:
        """Get value for action based on message-passed embeddings"""
        if not self.nodes:
            return 0.0
        
        # Find similar episodes
        query = np.array([pos[0] / self.width, pos[1] / self.height, action / 3.0, 0])
        
        value = 0.0
        total_weight = 0.0
        
        for node in self.nodes:
            # Distance in embedding space
            dist = np.linalg.norm(query[:3] - node.embedding[:3])
            weight = np.exp(-dist * 5)
            
            # Use the reward component of embedding
            value += weight * node.embedding[3]
            total_weight += weight
        
        if total_weight > 0:
            return value / total_weight
        return 0.0
    
    def navigate(self, goal: Tuple[int, int], max_steps: int = 3000) -> Dict:
        """Navigate with message passing"""
        print(f"\nSimple Message Passing ({self.num_hops}-hop)")
        print(f"Start: {self.position}, Goal: {goal}")
        
        steps = 0
        start_time = time.time()
        
        while self.position != goal and steps < max_steps:
            if steps % 500 == 0 and steps > 0:
                dist = abs(self.position[0] - goal[0]) + abs(self.position[1] - goal[1])
                print(f"Step {steps}: pos={self.position}, dist={dist}, "
                      f"nodes={len(self.nodes)}, edges={sum(len(v) for v in self.adjacency.values())//2}")
            
            # Message pass every 10 steps
            if steps % 10 == 0 and self.nodes:
                self.message_pass()
            
            # Evaluate actions
            action_values = []
            for action in range(4):
                value = self.get_action_value(self.position, action)
                action_values.append(value)
            
            # Add exploration noise
            values = np.array(action_values)
            values += np.random.normal(0, 0.1, 4)
            
            # Softmax
            probs = np.exp(values)
            probs = probs / probs.sum()
            action = np.random.choice(4, p=probs)
            
            # Execute
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
            new_pos = (self.position[0] + dx, self.position[1] + dy)
            
            # Check valid
            if (0 <= new_pos[0] < self.width and 
                0 <= new_pos[1] < self.height and
                self.maze[new_pos[1], new_pos[0]] == 0):
                reward = 1.0
                self.position = new_pos
                self.visited.add(new_pos)
                self.moves += 1
            else:
                reward = -1.0
                self.wall_hits += 1
            
            # Add episode
            self.add_episode(self.position, action, reward)
            
            steps += 1
        
        success = self.position == goal
        elapsed = time.time() - start_time
        
        print(f"Complete: success={success}, steps={steps}, wall_hits={self.wall_hits}")
        print(f"Nodes: {len(self.nodes)}, Edges: {sum(len(v) for v in self.adjacency.values())//2}")
        
        return {
            'success': success,
            'steps': steps,
            'wall_hits': self.wall_hits,
            'time': elapsed
        }


def test_simple_message_passing():
    """Test message passing effect"""
    # Small maze
    np.random.seed(42)
    size = 20
    maze = np.ones((size, size), dtype=int)
    
    def carve(x, y):
        maze[y, x] = 0
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        np.random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < size-1 and 0 < ny < size-1 and maze[ny, nx] == 1:
                maze[y + dy//2, x + dx//2] = 0
                maze[ny, nx] = 0
                carve(nx, ny)
    
    carve(1, 1)
    maze[size-2, size-2] = 0
    
    print("="*60)
    print("MESSAGE PASSING EFFECT TEST")
    print("="*60)
    
    # Test different hop counts
    for num_hops in [0, 1, 2, 3]:
        nav = SimpleMessagePassing(maze, num_hops=num_hops)
        result = nav.navigate((size-2, size-2))
        print(f"\n{num_hops}-hop: Steps={result['steps']}, "
              f"Wall hits={result['wall_hits']}, "
              f"Success={result['success']}")


if __name__ == "__main__":
    test_simple_message_passing()