#!/usr/bin/env python3
"""
Optimized True geDIG Navigator
==============================

In-memory graph with batch database updates.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
import time
from collections import defaultdict
import sqlite3
import os

class OptimizedTrueGeDIG:
    """Optimized navigator with in-memory graph and batch DB updates"""
    
    def __init__(self, maze: np.ndarray, batch_size: int = 100):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.goal = (self.width-2, self.height-2)
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        
        # In-memory storage
        self.episodes = []
        self.graph = defaultdict(list)  # {ep_id: [(neighbor_id, weight), ...]}
        self.batch_size = batch_size
        self.pending_episodes = []
        
        # Visual memory
        self.visual_memory = {}
        self._update_visual_memory(1, 1)
        
        # Statistics
        self.wall_hits = 0
        self.moves = 0
        self.hop_selections = {'1-hop': 0, '2-hop': 0, '3-hop': 0}
        
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
            10.0 if reached_goal else 0.0
        ])
        
        return embedding
    
    def add_episode(self, pos: Tuple[int, int], action: str, 
                   result: str, reached_goal: bool):
        """Add episode and update in-memory graph"""
        embedding = self.create_episode_embedding(pos, action, result, reached_goal)
        
        episode = {
            'id': len(self.episodes),
            'pos': pos,
            'action': action,
            'result': result,
            'reached_goal': reached_goal,
            'embedding': embedding
        }
        
        # Update graph connections
        for other in self.episodes:
            # Spatial distance
            spatial_dist = abs(pos[0] - other['pos'][0]) + abs(pos[1] - other['pos'][1])
            
            if spatial_dist <= 3:
                # Embedding similarity
                emb_sim = np.dot(embedding, other['embedding']) / (
                    np.linalg.norm(embedding) * np.linalg.norm(other['embedding']) + 1e-8
                )
                
                # Edge weight
                weight = emb_sim * np.exp(-spatial_dist * 0.3)
                
                if weight > 0.1:
                    self.graph[episode['id']].append((other['id'], weight))
                    self.graph[other['id']].append((episode['id'], weight))
        
        self.episodes.append(episode)
    
    def get_n_hop_neighbors(self, start_pos: Tuple[int, int], n_hops: int) -> List[Dict]:
        """Get n-hop neighbors from in-memory graph"""
        if not self.episodes:
            return []
        
        # Find closest episode
        min_dist = float('inf')
        start_id = None
        
        for ep in self.episodes:
            dist = abs(ep['pos'][0] - start_pos[0]) + abs(ep['pos'][1] - start_pos[1])
            if dist < min_dist:
                min_dist = dist
                start_id = ep['id']
        
        if start_id is None:
            return []
        
        # BFS for n hops
        visited = {start_id}
        current_layer = {start_id}
        result_episodes = []
        
        for hop in range(n_hops):
            if not current_layer:
                break
            
            next_layer = set()
            for ep_id in current_layer:
                for neighbor_id, weight in self.graph[ep_id]:
                    if neighbor_id not in visited:
                        next_layer.add(neighbor_id)
                        ep = self.episodes[neighbor_id]
                        result_episodes.append({
                            'id': neighbor_id,
                            'pos': ep['pos'],
                            'embedding': ep['embedding'],
                            'reached_goal': ep['reached_goal'],
                            'weight': weight,
                            'hop': hop + 1
                        })
            
            visited.update(next_layer)
            current_layer = next_layer
        
        return result_episodes
    
    def gedig_message_pass(self, episodes: List[Dict]) -> Dict[int, np.ndarray]:
        """Fast in-memory message passing"""
        if not episodes:
            return {}
        
        embeddings = {ep['id']: ep['embedding'].copy() for ep in episodes}
        ep_ids = set(embeddings.keys())
        
        # Message passing rounds
        for _ in range(3):
            new_embeddings = {}
            
            for ep_id in embeddings:
                # Get neighbors in the subgraph
                messages = []
                weights = []
                
                for neighbor_id, weight in self.graph[ep_id]:
                    if neighbor_id in ep_ids:
                        messages.append(embeddings[neighbor_id])
                        weights.append(weight)
                
                if messages:
                    weights = np.array(weights)
                    weights = weights / weights.sum()
                    
                    # Weighted average
                    avg_message = np.sum([m * w for m, w in zip(messages, weights)], axis=0)
                    
                    # Mix
                    new_emb = embeddings[ep_id].copy()
                    new_emb[5] = 0.3 * embeddings[ep_id][5] + 0.7 * avg_message[5]
                    new_emb[:5] = 0.7 * embeddings[ep_id][:5] + 0.3 * avg_message[:5]
                    
                    new_embeddings[ep_id] = new_emb
                else:
                    new_embeddings[ep_id] = embeddings[ep_id]
            
            embeddings = new_embeddings
        
        return embeddings
    
    def evaluate_action(self, pos: Tuple[int, int], action: str) -> Tuple[float, int]:
        """Evaluate action using true geDIG"""
        dx, dy = {'up': (0, -1), 'right': (1, 0), 
                 'down': (0, 1), 'left': (-1, 0)}[action]
        next_pos = (pos[0] + dx, pos[1] + dy)
        
        best_score = -float('inf')
        best_hop = 1
        
        for n_hops in [1, 2, 3]:
            episodes = self.get_n_hop_neighbors(next_pos, n_hops)
            
            if not episodes:
                continue
            
            # Message passing
            updated = self.gedig_message_pass(episodes)
            
            # Score calculation
            score = 0.0
            total_weight = 0.0
            
            for ep in episodes:
                if ep['id'] in updated:
                    goal_signal = updated[ep['id']][5]
                    success_signal = updated[ep['id']][3]
                    
                    graph_weight = ep.get('weight', 1.0)
                    hop_penalty = 0.8 ** (ep.get('hop', 1) - 1)
                    
                    ep_score = (goal_signal * 0.6 + success_signal * 0.4) * graph_weight * hop_penalty
                    score += ep_score
                    total_weight += graph_weight
            
            if total_weight > 0:
                score /= total_weight
            
            score += n_hops * 0.02
            
            if score > best_score:
                best_score = score
                best_hop = n_hops
        
        self.hop_selections[f'{best_hop}-hop'] += 1
        return best_score, best_hop
    
    def decide_action(self) -> str:
        """Decide action"""
        visual = self.visual_memory.get(self.position, {})
        
        action_scores = {}
        
        for action in ['up', 'right', 'down', 'left']:
            if visual.get(action) == 'wall':
                action_scores[action] = -10.0
                continue
            
            score, _ = self.evaluate_action(self.position, action)
            action_scores[action] = score
            
            # Exploration
            dx, dy = {'up': (0, -1), 'right': (1, 0), 
                     'down': (0, 1), 'left': (-1, 0)}[action]
            next_pos = (self.position[0] + dx, self.position[1] + dy)
            
            if next_pos not in self.visited:
                action_scores[action] += 2.0
        
        if not action_scores:
            return 'up'
        
        # Softmax
        actions = list(action_scores.keys())
        values = np.array(list(action_scores.values()))
        
        temperature = 0.3
        exp_values = np.exp((values - values.max()) / temperature)
        probs = exp_values / exp_values.sum()
        
        return np.random.choice(actions, p=probs)
    
    def navigate(self, max_steps: int = 5000) -> Dict:
        """Navigate maze"""
        print(f"\nOptimized True geDIG Navigation")
        print(f"Start: {self.position}, Goal: {self.goal}")
        print(f"Maze size: {self.width}×{self.height}")
        
        steps = 0
        start_time = time.time()
        
        while self.position != self.goal and steps < max_steps:
            if steps % 500 == 0 and steps > 0:
                dist = abs(self.position[0] - self.goal[0]) + abs(self.position[1] - self.goal[1])
                coverage = len(self.visited) / (self.width * self.height) * 100
                
                # Graph stats
                total_edges = sum(len(neighbors) for neighbors in self.graph.values()) // 2
                
                print(f"Step {steps}: pos={self.position}, dist={dist}, "
                      f"coverage={coverage:.1f}%, episodes={len(self.episodes)}, "
                      f"edges={total_edges}")
                
                # Hop usage
                total = sum(self.hop_selections.values())
                if total > 0:
                    print("  Hop usage:", end=" ")
                    for hop, count in self.hop_selections.items():
                        print(f"{hop}: {count/total*100:.1f}%", end=" ")
                    print()
            
            # Navigate
            action = self.decide_action()
            
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
                self.moves += 1
                self._update_visual_memory(new_pos[0], new_pos[1])
                
                if new_pos == self.goal:
                    reached_goal = True
            else:
                self.wall_hits += 1
            
            self.add_episode(old_pos, action, result, reached_goal)
            steps += 1
        
        elapsed = time.time() - start_time
        success = self.position == self.goal
        
        # Final stats
        total_edges = sum(len(neighbors) for neighbors in self.graph.values()) // 2
        
        print(f"\nComplete! Success: {success}")
        print(f"Steps: {steps}, Wall hits: {self.wall_hits}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Total episodes: {len(self.episodes)}, Total edges: {total_edges}")
        print(f"Average degree: {total_edges * 2 / len(self.episodes):.1f}")
        
        # Hop distribution
        print(f"\nFinal hop selection distribution:")
        total = sum(self.hop_selections.values())
        if total > 0:
            for hop, count in self.hop_selections.items():
                print(f"  {hop}: {count} ({count/total*100:.1f}%)")
        
        return {
            'success': success,
            'steps': steps,
            'wall_hits': self.wall_hits,
            'time': elapsed,
            'path_length': len(self.path),
            'hop_selections': dict(self.hop_selections),
            'total_episodes': len(self.episodes),
            'total_edges': total_edges
        }


def test_50x50():
    """Test on 50x50 maze"""
    from pure_episodic_navigator import create_complex_maze, visualize_maze_with_path
    
    print("="*70)
    print("OPTIMIZED TRUE geDIG - 50×50 TEST")
    print("="*70)
    
    size = 50
    maze = create_complex_maze(size, seed=42)
    nav = OptimizedTrueGeDIG(maze)
    
    result = nav.navigate(max_steps=10000)
    
    if result['success']:
        efficiency = result['steps'] / (2 * (size - 2))
        print(f"\n✓ SUCCESS!")
        print(f"Efficiency: {efficiency:.2f}x optimal")
        
        # Save visualization
        os.makedirs('visualizations', exist_ok=True)
        visualize_maze_with_path(
            maze, nav.path,
            f'visualizations/optimized_true_gedig_50x50.png'
        )
    else:
        print(f"\n✗ Failed")
        print(f"Explored {len(nav.visited)} cells ({len(nav.visited)/(size*size)*100:.1f}%)")


if __name__ == "__main__":
    test_50x50()