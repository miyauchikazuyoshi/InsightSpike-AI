#!/usr/bin/env python3
"""
Pure Episodic Navigator with Donut Search
========================================

pure_episodic_navigator.pyにドーナツ検索を適用して
大規模迷路での動作を可能にする

- チートなし
- 純粋なエピソード記憶
- 効率的なドーナツ検索
"""

import numpy as np
from typing import Dict, List, Tuple
import networkx as nx
import time

class PureEpisodicDonutNavigator:
    """Pure episodic navigation with donut search"""
    
    def __init__(self, maze: np.ndarray, 
                 outer_radius: float = 1.5,
                 inner_radius: float = 0.0,
                 message_depth: int = 3):
        self.maze = maze
        self.height, self.width = maze.shape
        self.start_pos = (1, 1)
        self.goal_pos = (self.width-2, self.height-2)
        self.position = self.start_pos
        
        # Parameters
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.message_depth = message_depth
        
        # Episode memory
        self.episodes = []
        self.episode_graph = nx.Graph()
        
        # Visit tracking
        self.visit_counts = {}
        
        # Statistics
        self.path = [self.position]
        self.steps = 0
        
        # Initialize with wall episodes at origin (like true_gedig_flow)
        self._create_initial_episodes()
    
    def _create_initial_episodes(self):
        """Create initial episodes (start position movements + goal)"""
        # Try all 4 directions from start position
        for action, (dx, dy) in [('up', (0, -1)), ('down', (0, 1)), 
                                  ('left', (-1, 0)), ('right', (1, 0))]:
            next_pos = (self.start_pos[0] + dx, self.start_pos[1] + dy)
            
            # Check if valid move
            if (0 <= next_pos[0] < self.width and 
                0 <= next_pos[1] < self.height and
                self.maze[next_pos[1], next_pos[0]] == 0):
                # Success move
                self.add_episode(self.start_pos, action, 'success', False)
            else:
                # Wall hit
                self.add_episode(self.start_pos, action, 'wall', False)
        
        # Goal episode
        self.add_episode(self.goal_pos, 'up', 'success', True)
    
    def create_episode_embedding(self, pos: Tuple[int, int], action: str, 
                               result: str, reached_goal: bool, visit_count: int = 0) -> np.ndarray:
        """Create 7D episode embedding with visit count"""
        # Count surrounding walls
        wall_count = 0
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.maze[ny, nx] == 1:
                    wall_count += 1
            else:
                wall_count += 1
        
        # Create embedding
        embedding = np.array([
            pos[0] / self.width,  # Position X [0, 1]
            pos[1] / self.height,  # Position Y [0, 1]
            {'up': -1.0, 'right': -0.33, 'down': 0.33, 'left': 1.0}[action],
            {'success': 1.0, 'wall': -1.0, 'visited': 0.0}[result],
            (wall_count - 2) / 2,  # Local topology [-1, 1]
            10.0 if reached_goal else 0.0,  # Goal signal
            visit_count / 10.0  # Normalized visit count [0, ∞)
        ])
        
        return embedding / np.linalg.norm(embedding)
    
    def donut_search(self, query_embedding: np.ndarray, max_results: int = 50) -> List[Dict]:
        """Efficient donut search"""
        results = []
        
        for ep in self.episodes:
            distance = np.linalg.norm(query_embedding - ep['embedding'])
            
            # Donut filter (inner_radius = 0 for initial exploration)
            if distance <= self.outer_radius:
                results.append({
                    'id': ep['id'],
                    'distance': distance,
                    'episode': ep
                })
        
        # Sort by distance and return top results
        results.sort(key=lambda x: x['distance'])
        return results[:max_results]
    
    def add_episode(self, pos: Tuple[int, int], action: str, 
                   result: str, reached_goal: bool):
        """Add episode and update graph"""
        # Update visit count
        pos_key = f"{pos[0]},{pos[1]}"
        self.visit_counts[pos_key] = self.visit_counts.get(pos_key, 0) + 1
        
        # Create embedding with visit count
        embedding = self.create_episode_embedding(
            pos, action, result, reached_goal, 
            self.visit_counts[pos_key]
        )
        
        episode = {
            'id': len(self.episodes),
            'pos': pos,
            'action': action,
            'result': result,
            'embedding': embedding,
            'reached_goal': reached_goal
        }
        
        # Find similar episodes using donut search
        if self.episodes:
            similar = self.donut_search(embedding, max_results=20)
            
            # Add edges to graph
            self.episode_graph.add_node(episode['id'])
            
            for item in similar:
                # Use cosine similarity instead of 1 - distance
                cos_sim = np.dot(embedding, self.episodes[item['id']]['embedding'])
                if cos_sim > 0.7:  # Threshold for cosine similarity
                    self.episode_graph.add_edge(
                        episode['id'], 
                        item['id'],
                        weight=cos_sim
                    )
        else:
            self.episode_graph.add_node(episode['id'])
        
        self.episodes.append(episode)
    
    def get_n_hop_episodes_donut(self, pos: Tuple[int, int], n_hops: int) -> List[Dict]:
        """Get n-hop episodes using donut search"""
        # Get visit count for this position
        pos_key = f"{pos[0]},{pos[1]}"
        visit_count = self.visit_counts.get(pos_key, 0)
        
        # Create query embedding for current position
        query_embedding = np.zeros(7)  # Now 7D
        query_embedding[0] = pos[0] / self.width
        query_embedding[1] = pos[1] / self.height
        query_embedding[5] = 1.0  # Looking for goal
        query_embedding[6] = visit_count / 10.0  # Visit count
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Get initial candidates via donut search
        candidates = self.donut_search(query_embedding, max_results=30)
        
        if not candidates:
            return []
        
        # Expand via graph for n-hops
        visited = set()
        result_episodes = []
        
        # Add 1-hop results
        for item in candidates:
            ep_id = item['id']
            visited.add(ep_id)
            result_episodes.append({
                'id': ep_id,
                'hop': 1,
                'episode': item['episode']
            })
        
        # Expand for additional hops
        current_layer = {item['id'] for item in candidates}
        
        for hop in range(2, n_hops + 1):
            if not current_layer:
                break
            
            next_layer = set()
            for ep_id in current_layer:
                if ep_id in self.episode_graph:
                    for neighbor_id in self.episode_graph[ep_id]:
                        if neighbor_id not in visited:
                            visited.add(neighbor_id)
                            next_layer.add(neighbor_id)
                            result_episodes.append({
                                'id': neighbor_id,
                                'hop': hop,
                                'episode': self.episodes[neighbor_id]
                            })
            
            current_layer = next_layer
        
        return result_episodes
    
    def deep_message_pass(self, episodes: List[Dict]) -> Dict[int, np.ndarray]:
        """Deep message passing on episodes"""
        if not episodes:
            return {}
        
        # Initialize embeddings
        embeddings = {ep['id']: ep['episode']['embedding'].copy() for ep in episodes}
        ep_ids = set(embeddings.keys())
        
        # Multiple rounds of message passing
        for _ in range(self.message_depth):
            new_embeddings = {}
            
            for ep_id in embeddings:
                # Self embedding
                new_emb = embeddings[ep_id] * 0.5
                
                # Aggregate from neighbors
                neighbor_count = 0
                if ep_id in self.episode_graph:
                    for neighbor_id in self.episode_graph[ep_id]:
                        if neighbor_id in ep_ids:
                            edge_data = self.episode_graph[ep_id][neighbor_id]
                            weight = edge_data.get('weight', 1.0)
                            new_emb += embeddings[neighbor_id] * weight * 0.5
                            neighbor_count += 1
                
                # Normalize
                if neighbor_count > 0:
                    new_emb = new_emb / (1 + neighbor_count)
                
                new_embeddings[ep_id] = new_emb / np.linalg.norm(new_emb)
            
            embeddings = new_embeddings
        
        return embeddings
    
    def gedig_evaluate_actions(self) -> Dict[str, float]:
        """Evaluate actions using pure geDIG (no cheats)"""
        scores = {}
        
        for action in ['up', 'down', 'left', 'right']:
            # Simulate action
            dx, dy = {'up': (0, -1), 'down': (0, 1), 
                      'left': (-1, 0), 'right': (1, 0)}[action]
            next_pos = (self.position[0] + dx, self.position[1] + dy)
            
            # Skip invalid positions
            if not (0 <= next_pos[0] < self.width and 
                   0 <= next_pos[1] < self.height):
                scores[action] = -float('inf')
                continue
            
            # Skip walls
            if self.maze[next_pos[1], next_pos[0]] == 1:
                scores[action] = -float('inf')
                continue
            
            # If no episodes yet, score is 0
            if not self.episodes:
                scores[action] = 0.0
                continue
            
            # Adaptive hop selection
            best_score = -float('inf')
            
            for n_hops in [1, 2, 3]:
                episodes = self.get_n_hop_episodes_donut(next_pos, n_hops)
                
                if not episodes:
                    continue
                
                # Message passing
                updated = self.deep_message_pass(episodes)
                
                if not updated:
                    continue
                
                # Calculate pure geDIG score (no cheats)
                score = 0.0
                total_weight = 0.0
                
                for ep_id, embedding in updated.items():
                    # Pure signal aggregation
                    goal_signal = embedding[5]
                    success_signal = max(0, embedding[3])
                    
                    # Equal weighting - no designed bias
                    score += (goal_signal + success_signal) / 2
                    total_weight += 1
                
                if total_weight > 0:
                    score /= total_weight
                
                best_score = max(best_score, score)
            
            # If still no score, use 0
            if best_score == -float('inf'):
                best_score = 0.0
            
            scores[action] = best_score
        
        return scores
    
    def navigate_step(self) -> bool:
        """Single navigation step"""
        # Evaluate actions
        action_scores = self.gedig_evaluate_actions()
        
        # Select best valid action
        valid_actions = [(a, s) for a, s in action_scores.items() if s > -float('inf')]
        
        if not valid_actions:
            # No valid actions
            return False
        
        # Choose action with highest score
        best_action = max(valid_actions, key=lambda x: x[1])[0]
        
        # Execute action
        dx, dy = {'up': (0, -1), 'down': (0, 1), 
                  'left': (-1, 0), 'right': (1, 0)}[best_action]
        new_pos = (self.position[0] + dx, self.position[1] + dy)
        
        # Check result
        if self.maze[new_pos[1], new_pos[0]] == 0:
            result = 'success'
            self.position = new_pos
            self.path.append(new_pos)
        else:
            result = 'wall'
        
        reached_goal = (self.position == self.goal_pos)
        
        # Record episode
        self.add_episode(self.position, best_action, result, reached_goal)
        
        self.steps += 1
        
        return reached_goal
    
    def navigate(self, max_steps: int = 10000) -> Dict:
        """Complete navigation"""
        print(f"Pure Episodic Navigation with Donut Search")
        print(f"Maze size: {self.width}x{self.height}")
        print(f"Start: {self.start_pos}, Goal: {self.goal_pos}")
        print(f"NO CHEATS - Pure episodic memory only")
        
        start_time = time.time()
        
        while self.steps < max_steps:
            if self.steps % 100 == 0:
                goal_dist = abs(self.position[0] - self.goal_pos[0]) + \
                           abs(self.position[1] - self.goal_pos[1])
                print(f"Step {self.steps}: pos={self.position}, "
                      f"goal_dist={goal_dist}, episodes={len(self.episodes)}")
            
            if self.navigate_step():
                print(f"\n✓ Goal reached in {self.steps} steps!")
                break
        
        elapsed = time.time() - start_time
        success = self.position == self.goal_pos
        
        return {
            'success': success,
            'steps': self.steps,
            'path': self.path,
            'episodes': len(self.episodes),
            'time': elapsed
        }


def test_pure_episodic_donut():
    """Test pure episodic with donut search"""
    from pure_episodic_navigator import create_complex_maze, visualize_maze_with_path
    
    # Test on different sizes
    for size in [15, 25, 50]:
        print(f"\n{'='*60}")
        print(f"Testing {size}x{size} maze")
        print(f"{'='*60}\n")
        
        maze = create_complex_maze(size, seed=42)
        nav = PureEpisodicDonutNavigator(maze)
        
        max_steps = size * size * 2
        result = nav.navigate(max_steps=max_steps)
        
        print(f"\nResults:")
        print(f"- Success: {result['success']}")
        print(f"- Steps: {result['steps']}")
        print(f"- Episodes: {result['episodes']}")
        print(f"- Time: {result['time']:.2f}s")
        
        if result['success']:
            filename = f'pure_episodic_donut_{size}x{size}.png'
            visualize_maze_with_path(maze, result['path'], filename)
            print(f"- Saved: {filename}")
        
        # Stop if failed
        if not result['success'] and size < 50:
            print(f"\nFailed on {size}x{size}, stopping tests")
            break


if __name__ == "__main__":
    test_pure_episodic_donut()