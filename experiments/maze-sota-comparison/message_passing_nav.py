#!/usr/bin/env python3
"""
Message Passing Navigator
=========================

Uses message passing between episodes to generate insight vectors
for decision making.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import matplotlib.pyplot as plt

@dataclass
class Episode:
    """Episode with features and connections"""
    id: int
    position: Tuple[int, int]
    action: str
    success: bool
    features: np.ndarray
    embedding: Optional[np.ndarray] = None  # Will be updated by message passing

class MessagePassingNavigator:
    """Navigator using GNN-style message passing for decisions"""
    
    def __init__(self, maze: np.ndarray, num_layers: int = 3):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        
        # Episode memory and graph
        self.episodes: List[Episode] = []
        self.episode_graph = nx.Graph()
        self.episode_id_counter = 0
        
        # Message passing parameters
        self.num_layers = num_layers
        self.embedding_dim = 8
        
        # Statistics
        self.wall_hits = 0
        self.moves = 0
        
        # Initialize random weight matrices for message passing
        np.random.seed(42)
        self.W_msg = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.1
        self.W_update = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.1
        self.W_combine = np.random.randn(self.embedding_dim, self.embedding_dim * 2) * 0.1
        
    def _create_features(self, pos: Tuple[int, int], action: str) -> np.ndarray:
        """Create feature vector for position-action pair"""
        # Normalize position
        norm_x = pos[0] / self.width
        norm_y = pos[1] / self.height
        
        # Action encoding
        action_vec = np.zeros(4)
        action_idx = ['up', 'right', 'down', 'left'].index(action)
        action_vec[action_idx] = 1.0
        
        # Combine into feature vector
        features = np.concatenate([
            [norm_x, norm_y],
            action_vec
        ])
        
        return features
    
    def _add_episode(self, pos: Tuple[int, int], action: str, success: bool):
        """Add episode and update graph"""
        features = self._create_features(pos, action)
        
        # Initial embedding is just padded features
        initial_embedding = np.zeros(self.embedding_dim)
        initial_embedding[:len(features)] = features
        
        episode = Episode(
            id=self.episode_id_counter,
            position=pos,
            action=action,
            success=success,
            features=features,
            embedding=initial_embedding
        )
        
        self.episodes.append(episode)
        self.episode_graph.add_node(self.episode_id_counter, episode=episode)
        
        # Connect to similar episodes
        for other_ep in self.episodes[:-1]:
            # Spatial proximity
            dist = abs(pos[0] - other_ep.position[0]) + abs(pos[1] - other_ep.position[1])
            if dist <= 2:  # Connect nearby episodes
                weight = 1.0 / (1.0 + dist)
                self.episode_graph.add_edge(
                    self.episode_id_counter, 
                    other_ep.id,
                    weight=weight
                )
        
        self.episode_id_counter += 1
    
    def _message_passing(self):
        """Run message passing to update episode embeddings"""
        if not self.episodes:
            return
        
        # Initialize embeddings
        embeddings = {ep.id: ep.embedding.copy() for ep in self.episodes}
        
        # Run message passing layers
        for layer in range(self.num_layers):
            new_embeddings = {}
            
            for node_id in self.episode_graph.nodes():
                # Get current embedding
                h_i = embeddings[node_id]
                
                # Aggregate messages from neighbors
                messages = []
                for neighbor_id in self.episode_graph.neighbors(node_id):
                    h_j = embeddings[neighbor_id]
                    weight = self.episode_graph[node_id][neighbor_id]['weight']
                    
                    # Message: weighted neighbor embedding transformed
                    msg = weight * np.tanh(np.dot(self.W_msg, h_j))
                    messages.append(msg)
                
                if messages:
                    # Aggregate messages (mean)
                    agg_msg = np.mean(messages, axis=0)
                    
                    # Combine with self embedding
                    combined = np.concatenate([h_i, agg_msg])
                    new_h = np.tanh(np.dot(self.W_combine, combined))
                else:
                    # No neighbors - self update only
                    new_h = np.tanh(np.dot(self.W_update, h_i))
                
                new_embeddings[node_id] = new_h
            
            embeddings = new_embeddings
        
        # Update episode embeddings
        for ep in self.episodes:
            ep.embedding = embeddings[ep.id]
    
    def _generate_insight_vector(self, pos: Tuple[int, int], action: str) -> np.ndarray:
        """Generate insight vector for position-action using message passing results"""
        # Create query features
        query_features = self._create_features(pos, action)
        
        if not self.episodes:
            # No experience yet
            insight = np.zeros(self.embedding_dim)
            insight[:len(query_features)] = query_features
            return insight
        
        # Find relevant episodes
        relevant_embeddings = []
        relevance_weights = []
        
        for ep in self.episodes:
            # Similarity based on position and action
            pos_dist = abs(pos[0] - ep.position[0]) + abs(pos[1] - ep.position[1])
            same_action = (action == ep.action)
            
            relevance = np.exp(-pos_dist * 0.5)
            if same_action:
                relevance *= 2.0
            
            if relevance > 0.1:  # Threshold
                relevant_embeddings.append(ep.embedding)
                relevance_weights.append(relevance)
                
                # Extra weight for successful episodes
                if ep.success:
                    relevance_weights[-1] *= 1.5
        
        if relevant_embeddings:
            # Weighted average of relevant embeddings
            weights = np.array(relevance_weights)
            weights = weights / weights.sum()
            
            insight = np.zeros(self.embedding_dim)
            for w, emb in zip(weights, relevant_embeddings):
                insight += w * emb
        else:
            # No relevant experience - use query features
            insight = np.zeros(self.embedding_dim)
            insight[:len(query_features)] = query_features
        
        return insight
    
    def _evaluate_action(self, pos: Tuple[int, int], action: str) -> float:
        """Evaluate action using insight vector"""
        insight = self._generate_insight_vector(pos, action)
        
        # Simple evaluation: higher norm = better
        # (In practice, could use a learned value function)
        base_value = np.linalg.norm(insight)
        
        # Add exploration bonus for unvisited positions
        dx, dy = {'up': (0, -1), 'right': (1, 0), 
                  'down': (0, 1), 'left': (-1, 0)}[action]
        next_pos = (pos[0] + dx, pos[1] + dy)
        
        if next_pos not in self.visited:
            base_value += 0.5
        
        return base_value
    
    def decide_action(self, run_message_passing: bool = True) -> str:
        """Decide action using message-passed insight vectors"""
        # Run message passing to update embeddings (optional)
        if run_message_passing and len(self.episodes) > 0:
            self._message_passing()
        
        # Evaluate all actions
        action_values = {}
        for action in ['up', 'right', 'down', 'left']:
            value = self._evaluate_action(self.position, action)
            action_values[action] = value
        
        # Softmax selection
        actions = list(action_values.keys())
        values = np.array(list(action_values.values()))
        
        # Temperature
        temperature = 0.5
        exp_values = np.exp(values / temperature)
        probs = exp_values / exp_values.sum()
        
        return np.random.choice(actions, p=probs)
    
    def navigate(self, goal: Tuple[int, int], max_steps: int = 1000) -> Dict:
        """Navigate using message passing insights"""
        print(f"\nMessage Passing Navigation")
        print(f"Start: {self.position}, Goal: {goal}")
        print(f"Message passing layers: {self.num_layers}")
        
        steps = 0
        start_time = time.time()
        
        while self.position != goal and steps < max_steps:
            # Progress report
            if steps % 100 == 0:
                dist = abs(self.position[0] - goal[0]) + abs(self.position[1] - goal[1])
                print(f"Step {steps}: pos={self.position}, dist={dist}, "
                      f"episodes={len(self.episodes)}, graph_edges={self.episode_graph.number_of_edges()}")
            
            # Decide action (run message passing only every 20 steps to avoid slowdown)
            run_mp = (steps % 20 == 0)
            action = self.decide_action(run_message_passing=run_mp)
            
            # Execute action
            old_pos = self.position
            dx, dy = {'up': (0, -1), 'right': (1, 0), 
                     'down': (0, 1), 'left': (-1, 0)}[action]
            new_pos = (self.position[0] + dx, self.position[1] + dy)
            
            # Check if valid move
            success = False
            if (0 <= new_pos[0] < self.width and 
                0 <= new_pos[1] < self.height and
                self.maze[new_pos[1], new_pos[0]] == 0):
                # Success
                success = True
                self.position = new_pos
                self.visited.add(new_pos)
                self.path.append(new_pos)
                self.moves += 1
            else:
                # Hit wall
                self.wall_hits += 1
            
            # Record episode
            self._add_episode(old_pos, action, success)
            
            steps += 1
        
        success = self.position == goal
        elapsed = time.time() - start_time
        
        print(f"\nComplete: success={success}, steps={steps}")
        print(f"Wall hits: {self.wall_hits}, Success rate: {self.moves/(self.moves+self.wall_hits)*100:.1f}%")
        print(f"Graph: {self.episode_graph.number_of_nodes()} nodes, {self.episode_graph.number_of_edges()} edges")
        
        return {
            'success': success,
            'steps': steps,
            'wall_hits': self.wall_hits,
            'moves': self.moves,
            'graph_nodes': self.episode_graph.number_of_nodes(),
            'graph_edges': self.episode_graph.number_of_edges(),
            'time': elapsed
        }


def test_message_passing():
    """Test message passing navigation"""
    
    # Create maze
    np.random.seed(42)
    size = 10  # Smaller maze for debugging
    maze = np.ones((size, size), dtype=int)
    
    # Maze generation
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
    print("MESSAGE PASSING NAVIGATION TEST")
    print("="*60)
    
    # Test different layer counts
    results = {}
    for num_layers in [1, 2, 3]:
        print(f"\n--- {num_layers}-layer Message Passing ---")
        nav = MessagePassingNavigator(maze, num_layers=num_layers)
        result = nav.navigate((size-2, size-2), max_steps=1000)
        results[f"{num_layers}-layer"] = result
        
        # Debug: check if stuck
        if not result['success'] and result['steps'] >= 1000:
            print(f"WARNING: {num_layers}-layer timed out at max steps")
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    print(f"\n{'Method':<10} {'Success':<8} {'Steps':<8} {'Wall Hits':<10} {'Graph Edges':<12}")
    print("-" * 48)
    
    for method, result in results.items():
        print(f"{method:<10} {str(result['success']):<8} {result['steps']:<8} "
              f"{result['wall_hits']:<10} {result['graph_edges']:<12}")
    
    # Visualize the episode graph (last run)
    if nav.episode_graph.number_of_nodes() > 0:
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(nav.episode_graph, k=2, iterations=50)
        
        # Color nodes by success
        node_colors = []
        for node_id in nav.episode_graph.nodes():
            ep = nav.episode_graph.nodes[node_id]['episode']
            node_colors.append('green' if ep.success else 'red')
        
        nx.draw(nav.episode_graph, pos, 
                node_color=node_colors,
                node_size=50,
                edge_color='gray',
                alpha=0.6,
                with_labels=False)
        
        plt.title(f"Episode Graph ({nav.episode_graph.number_of_nodes()} nodes, "
                  f"{nav.episode_graph.number_of_edges()} edges)")
        plt.tight_layout()
        plt.savefig('message_passing_graph.png', dpi=150)
        plt.close()
        
        print(f"\nEpisode graph saved to message_passing_graph.png")
    
    return results


if __name__ == "__main__":
    results = test_message_passing()