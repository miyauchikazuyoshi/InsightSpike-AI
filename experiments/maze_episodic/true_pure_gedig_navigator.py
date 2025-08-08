#!/usr/bin/env python3
"""
True Pure geDIG Navigator
========================

Implements actual geDIG = GED - IG minimization.
No cheating, no exploration bonus.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
import time
from collections import defaultdict
import networkx as nx

class TruePureGeDIGNavigator:
    """Navigator using true geDIG evaluation (GED - IG)"""
    
    def __init__(self, maze: np.ndarray):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.goal = (self.width-2, self.height-2)
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        
        # Episode storage
        self.episodes = []
        self.episode_graph = nx.Graph()
        
        # Visual memory
        self.visual_memory = {}
        self._update_visual_memory(1, 1)
        
        # Statistics
        self.wall_hits = 0
        self.gedig_calculations = 0
        
    def _update_visual_memory(self, x: int, y: int):
        """Update visual memory"""
        self.visual_memory[(x, y)] = {}
        for action, (dx, dy) in {'up': (0, -1), 'right': (1, 0), 
                                'down': (0, 1), 'left': (-1, 0)}.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                self.visual_memory[(x, y)][action] = 'path' if self.maze[ny, nx] == 0 else 'wall'
    
    def create_episode_embedding(self, pos: Tuple[int, int], action: str, 
                               result: str, reached_goal: bool) -> np.ndarray:
        """Create episode embedding"""
        visual = self.visual_memory.get(pos, {})
        wall_count = sum(1 for d in ['up', 'right', 'down', 'left']
                        if visual.get(d) == 'wall')
        
        embedding = np.array([
            pos[0] / self.width,
            pos[1] / self.height,
            {'up': -1.0, 'right': -0.33, 'down': 0.33, 'left': 1.0}[action],
            {'success': 1.0, 'wall': -1.0, 'visited': 0.0}[result],
            (wall_count - 2) / 2,
            1.0 if reached_goal else 0.0
        ], dtype=np.float32)
        
        return embedding
    
    def add_episode(self, pos: Tuple[int, int], action: str, 
                   result: str, reached_goal: bool):
        """Add episode and update graph"""
        embedding = self.create_episode_embedding(pos, action, result, reached_goal)
        
        episode = {
            'id': len(self.episodes),
            'pos': pos,
            'action': action,
            'result': result,
            'reached_goal': reached_goal,
            'embedding': embedding
        }
        
        # Add node to graph
        self.episode_graph.add_node(
            episode['id'],
            pos=pos,
            embedding=embedding,
            reached_goal=reached_goal
        )
        
        # Connect to spatially close episodes
        for other_id in range(len(self.episodes)):
            other = self.episodes[other_id]
            dist = abs(pos[0] - other['pos'][0]) + abs(pos[1] - other['pos'][1])
            
            if dist <= 3:  # Within 3 Manhattan distance
                # Edge weight based on similarity
                similarity = np.dot(embedding, other['embedding'])
                self.episode_graph.add_edge(episode['id'], other_id, weight=similarity)
        
        self.episodes.append(episode)
    
    def calculate_ged(self, g1: nx.Graph, g2: nx.Graph) -> float:
        """Calculate normalized Graph Edit Distance"""
        # Simple GED approximation
        nodes1, nodes2 = set(g1.nodes()), set(g2.nodes())
        edges1, edges2 = set(g1.edges()), set(g2.edges())
        
        node_diff = len(nodes1.symmetric_difference(nodes2))
        edge_diff = len(edges1.symmetric_difference(edges2))
        
        # Normalize by total size
        total_size = max(len(nodes1) + len(nodes2), 1)
        ged = (node_diff + edge_diff) / total_size
        
        return ged
    
    def calculate_ig(self, subgraph: nx.Graph) -> float:
        """Calculate Information Gain based on goal information"""
        if len(subgraph) == 0:
            return 0.0
        
        # Calculate entropy before (uniform)
        h_before = np.log(len(subgraph) + 1e-10)
        
        # Calculate entropy after (considering goal information)
        goal_count = sum(1 for n in subgraph.nodes() 
                        if subgraph.nodes[n].get('reached_goal', False))
        
        if goal_count == 0:
            # No goal information
            h_after = h_before
        else:
            # Goal information reduces entropy
            p_goal = goal_count / len(subgraph)
            p_no_goal = 1 - p_goal
            
            h_after = 0
            if p_goal > 0:
                h_after -= p_goal * np.log(p_goal + 1e-10)
            if p_no_goal > 0:
                h_after -= p_no_goal * np.log(p_no_goal + 1e-10)
        
        # Information gain
        ig = h_before - h_after
        return ig
    
    def get_local_subgraph(self, center_pos: Tuple[int, int], radius: int = 3) -> nx.Graph:
        """Get subgraph of episodes near a position"""
        subgraph_nodes = []
        
        for ep in self.episodes:
            dist = abs(ep['pos'][0] - center_pos[0]) + abs(ep['pos'][1] - center_pos[1])
            if dist <= radius:
                subgraph_nodes.append(ep['id'])
        
        return self.episode_graph.subgraph(subgraph_nodes).copy()
    
    def evaluate_action_gedig(self, pos: Tuple[int, int], action: str) -> float:
        """Evaluate action using true geDIG = GED - IG"""
        dx, dy = {'up': (0, -1), 'right': (1, 0), 
                 'down': (0, 1), 'left': (-1, 0)}[action]
        next_pos = (pos[0] + dx, pos[1] + dy)
        
        # Get current local graph
        current_graph = self.get_local_subgraph(pos)
        
        # Simulate what would happen after this action
        # Create hypothetical new episode
        hypothetical_result = 'success'  # Assume success
        hypothetical_goal = (next_pos == self.goal)
        
        # Create future graph (copy current and add new episode)
        future_graph = current_graph.copy()
        new_id = len(self.episodes)  # Hypothetical new ID
        
        future_graph.add_node(
            new_id,
            pos=next_pos,
            reached_goal=hypothetical_goal
        )
        
        # Connect to nearby nodes
        for node in current_graph.nodes():
            node_pos = self.episodes[node]['pos']
            dist = abs(next_pos[0] - node_pos[0]) + abs(next_pos[1] - node_pos[1])
            if dist <= 3:
                future_graph.add_edge(new_id, node)
        
        # Calculate GED and IG
        ged = self.calculate_ged(current_graph, future_graph)
        ig = self.calculate_ig(future_graph)
        
        # geDIG = GED - IG (we want to minimize this)
        gedig = ged - ig
        
        self.gedig_calculations += 1
        
        return gedig
    
    def decide_action(self) -> str:
        """Decide action by minimizing geDIG"""
        visual = self.visual_memory.get(self.position, {})
        
        action_gedigs = {}
        
        for action in ['up', 'right', 'down', 'left']:
            # Skip walls
            if visual.get(action) == 'wall':
                continue
            
            # Calculate geDIG for this action
            gedig = self.evaluate_action_gedig(self.position, action)
            action_gedigs[action] = gedig
        
        if not action_gedigs:
            return 'up'  # Fallback
        
        # Choose action with minimum geDIG
        # Add small random noise to break ties
        best_action = min(action_gedigs.items(), 
                         key=lambda x: x[1] + np.random.normal(0, 0.01))[0]
        
        return best_action
    
    def navigate(self, max_steps: int = 5000) -> Dict:
        """Navigate using true geDIG minimization"""
        print(f"\nTrue Pure geDIG Navigation")
        print(f"Start: {self.position}, Goal: {self.goal}")
        print(f"Maze size: {self.width}×{self.height}")
        print(f"Strategy: Minimize geDIG = GED - IG")
        
        steps = 0
        start_time = time.time()
        
        while self.position != self.goal and steps < max_steps:
            if steps % 100 == 0 and steps > 0:
                dist = abs(self.position[0] - self.goal[0]) + abs(self.position[1] - self.goal[1])
                coverage = len(self.visited) / (self.width * self.height) * 100
                print(f"Step {steps}: pos={self.position}, dist={dist}, "
                      f"coverage={coverage:.1f}%, episodes={len(self.episodes)}, "
                      f"gedig_calcs={self.gedig_calculations}")
            
            # Decide action
            action = self.decide_action()
            
            # Execute
            old_pos = self.position
            dx, dy = {'up': (0, -1), 'right': (1, 0), 
                     'down': (0, 1), 'left': (-1, 0)}[action]
            new_pos = (self.position[0] + dx, self.position[1] + dy)
            
            result = 'wall'
            reached_goal = False
            
            if (0 <= new_pos[0] < self.width and 
                0 <= new_pos[1] < self.height and
                self.maze[new_pos[1], new_pos[0]] == 0):
                
                if new_pos in self.visited:
                    result = 'visited'
                else:
                    result = 'success'
                
                self.position = new_pos
                self.visited.add(new_pos)
                self.path.append(new_pos)
                self._update_visual_memory(new_pos[0], new_pos[1])
                
                if new_pos == self.goal:
                    reached_goal = True
            else:
                self.wall_hits += 1
            
            self.add_episode(old_pos, action, result, reached_goal)
            steps += 1
        
        elapsed = time.time() - start_time
        success = self.position == self.goal
        
        print(f"\nComplete! Success: {success}")
        print(f"Steps: {steps}, Wall hits: {self.wall_hits}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Total geDIG calculations: {self.gedig_calculations}")
        print(f"Graph size: {self.episode_graph.number_of_nodes()} nodes, "
              f"{self.episode_graph.number_of_edges()} edges")
        
        return {
            'success': success,
            'steps': steps,
            'wall_hits': self.wall_hits,
            'time': elapsed,
            'path_length': len(self.path),
            'gedig_calculations': self.gedig_calculations
        }


def test_true_gedig():
    """Test true geDIG navigator"""
    from pure_episodic_navigator import create_complex_maze, visualize_maze_with_path
    
    print("="*70)
    print("TRUE PURE geDIG NAVIGATION TEST")
    print("="*70)
    
    # Test on small maze first
    size = 10
    maze = create_complex_maze(size, seed=42)
    
    nav = TruePureGeDIGNavigator(maze)
    result = nav.navigate(max_steps=1000)
    
    if result['success']:
        print(f"\n✓ SUCCESS!")
        print(f"Efficiency: {result['steps'] / (2 * (size - 2)):.2f}x optimal")
        
        visualize_maze_with_path(
            maze, nav.path,
            'true_pure_gedig_10x10.png'
        )
    
    # Analyze behavior
    print(f"\nBehavior Analysis:")
    print(f"- Used true geDIG = GED - IG minimization")
    print(f"- No exploration bonus or visit penalty")
    print(f"- Pure information-theoretic navigation")


if __name__ == "__main__":
    test_true_gedig()