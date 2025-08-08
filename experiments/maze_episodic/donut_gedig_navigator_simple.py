#!/usr/bin/env python3
"""
Donut-based True geDIG Navigator (Simplified)
============================================

Efficient geDIG implementation using donut search logic.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, NamedTuple
import time
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class NeighborEpisode:
    """Neighbor episode with distance and similarity"""
    id: int
    distance: float
    similarity: float
    embedding: np.ndarray

class DonutGeDIGNavigator:
    """Navigator using donut search for efficient geDIG evaluation"""
    
    def __init__(self, maze: np.ndarray, 
                 inner_radius: float = 0.1,   # Exclude very close
                 outer_radius: float = 0.5):  # Search radius
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.goal = (self.width-2, self.height-2)
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        
        # Donut search parameters
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        
        # Episode storage
        self.episodes = []
        self.graph = defaultdict(list)  # Adjacency list
        
        # Visual memory
        self.visual_memory = {}
        self._update_visual_memory(1, 1)
        
        # Statistics
        self.wall_hits = 0
        self.moves = 0
        self.hop_selections = {'1-hop': 0, '2-hop': 0, '3-hop': 0}
        self.search_times = []
        
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
        
        # 6D embedding (without normalization to preserve magnitudes)
        embedding = np.array([
            pos[0] / self.width,
            pos[1] / self.height,
            {'up': -1.0, 'right': -0.33, 'down': 0.33, 'left': 1.0}[action],
            {'success': 1.0, 'wall': -1.0, 'visited': 0.0}[result],
            (wall_count - 2) / 2,
            1.0 if reached_goal else 0.0  # Reduced goal signal
        ], dtype=np.float32)
        
        return embedding
    
    def donut_search(self, query_embedding: np.ndarray, 
                    exclude_ids: Optional[Set[int]] = None) -> List[NeighborEpisode]:
        """Perform donut search on episodes"""
        if not self.episodes:
            return []
        
        neighbors = []
        exclude_ids = exclude_ids or set()
        
        # Adaptive donut: disable inner radius for early episodes
        use_donut = len(self.episodes) > 100  # Enable donut after 100 episodes
        
        # Compute similarities with all episodes
        for ep in self.episodes:
            if ep['id'] in exclude_ids:
                continue
                
            # Compute similarity (cosine-like but without normalization)
            dot_product = np.dot(query_embedding, ep['embedding'])
            norm_query = np.linalg.norm(query_embedding)
            norm_ep = np.linalg.norm(ep['embedding'])
            
            if norm_query > 0 and norm_ep > 0:
                similarity = dot_product / (norm_query * norm_ep)
            else:
                similarity = 0.0
            
            # Convert to distance (1 - similarity)
            distance = 1.0 - similarity
            
            # Donut filter (or just outer radius check)
            if use_donut:
                # Full donut search
                if self.inner_radius < distance <= self.outer_radius:
                    neighbors.append(NeighborEpisode(
                        id=ep['id'],
                        distance=distance,
                        similarity=similarity,
                        embedding=ep['embedding']
                    ))
            else:
                # Only outer radius (sphere search)
                if distance <= self.outer_radius:
                    neighbors.append(NeighborEpisode(
                        id=ep['id'],
                        distance=distance,
                        similarity=similarity,
                        embedding=ep['embedding']
                    ))
        
        # Sort by similarity (descending)
        neighbors.sort(key=lambda x: x.similarity, reverse=True)
        
        # Return top 50
        return neighbors[:50]
    
    def add_episode(self, pos: Tuple[int, int], action: str, 
                   result: str, reached_goal: bool):
        """Add episode and update graph using donut search"""
        embedding = self.create_episode_embedding(pos, action, result, reached_goal)
        
        episode = {
            'id': len(self.episodes),
            'pos': pos,
            'action': action,
            'result': result,
            'reached_goal': reached_goal,
            'embedding': embedding
        }
        
        # Find neighbors using donut search
        neighbors = self.donut_search(embedding, exclude_ids={episode['id']})
        
        # Update graph with bidirectional edges
        for neighbor in neighbors:
            # Weight based on similarity
            weight = neighbor.similarity
            if weight > 0.5:  # Threshold for edge creation
                self.graph[episode['id']].append((neighbor.id, weight))
                self.graph[neighbor.id].append((episode['id'], weight))
        
        self.episodes.append(episode)
    
    def get_n_hop_neighbors_donut(self, query_pos: Tuple[int, int], n_hops: int) -> List[Dict]:
        """Get n-hop neighbors using donut search"""
        if not self.episodes:
            return []
        
        start_time = time.time()
        
        # Create query embedding (position-based)
        query_embedding = np.zeros(6, dtype=np.float32)
        query_embedding[0] = query_pos[0] / self.width
        query_embedding[1] = query_pos[1] / self.height
        # Other dimensions stay 0 for position-based query
        
        # Initial donut search
        initial_neighbors = self.donut_search(query_embedding)
        
        # Convert to result format
        visited = set()
        current_layer = set()
        result_episodes = []
        
        for neighbor in initial_neighbors:
            current_layer.add(neighbor.id)
            visited.add(neighbor.id)
            ep = self.episodes[neighbor.id]
            result_episodes.append({
                'id': neighbor.id,
                'pos': ep['pos'],
                'embedding': ep['embedding'],
                'reached_goal': ep['reached_goal'],
                'weight': neighbor.similarity,
                'hop': 1
            })
        
        # Multi-hop expansion using graph
        for hop in range(1, n_hops):
            if not current_layer:
                break
            
            next_layer = set()
            for ep_id in current_layer:
                for neighbor_id, weight in self.graph[ep_id]:
                    if neighbor_id not in visited:
                        next_layer.add(neighbor_id)
                        visited.add(neighbor_id)
                        ep = self.episodes[neighbor_id]
                        result_episodes.append({
                            'id': neighbor_id,
                            'pos': ep['pos'],
                            'embedding': ep['embedding'],
                            'reached_goal': ep['reached_goal'],
                            'weight': weight,
                            'hop': hop + 1
                        })
            
            current_layer = next_layer
        
        search_time = time.time() - start_time
        self.search_times.append(search_time)
        
        return result_episodes
    
    def gedig_message_pass(self, episodes: List[Dict]) -> Dict[int, np.ndarray]:
        """Message passing on selected episodes"""
        if not episodes:
            return {}
        
        embeddings = {ep['id']: ep['embedding'].copy() for ep in episodes}
        ep_ids = set(embeddings.keys())
        
        # Message passing rounds
        for _ in range(3):
            new_embeddings = {}
            
            for ep_id in embeddings:
                # Get neighbors from graph
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
                    
                    # Mix with different rates
                    new_emb = embeddings[ep_id].copy()
                    new_emb[5] = 0.3 * embeddings[ep_id][5] + 0.7 * avg_message[5]  # Goal
                    new_emb[:5] = 0.7 * embeddings[ep_id][:5] + 0.3 * avg_message[:5]
                    
                    new_embeddings[ep_id] = new_emb
                else:
                    new_embeddings[ep_id] = embeddings[ep_id]
            
            embeddings = new_embeddings
        
        return embeddings
    
    def evaluate_action(self, pos: Tuple[int, int], action: str) -> Tuple[float, int]:
        """Evaluate action using donut-based geDIG"""
        dx, dy = {'up': (0, -1), 'right': (1, 0), 
                 'down': (0, 1), 'left': (-1, 0)}[action]
        next_pos = (pos[0] + dx, pos[1] + dy)
        
        # Initial exploration: if no episodes, return small positive score
        if not self.episodes:
            # Prefer moves toward goal during initial exploration
            goal_dist_current = abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])
            goal_dist_next = abs(next_pos[0] - self.goal[0]) + abs(next_pos[1] - self.goal[1])
            
            # Small positive score, higher if moving toward goal
            score = 0.1
            if goal_dist_next < goal_dist_current:
                score += 0.2
            
            return score, 1
        
        best_score = -float('inf')
        best_hop = 1
        
        for n_hops in [1, 2, 3]:
            episodes = self.get_n_hop_neighbors_donut(next_pos, n_hops)
            
            if not episodes:
                continue
            
            # Message passing
            updated = self.gedig_message_pass(episodes)
            
            # Calculate score
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
        
        # If no episodes found near next_pos, use goal heuristic
        if best_score == -float('inf'):
            goal_dist_current = abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])
            goal_dist_next = abs(next_pos[0] - self.goal[0]) + abs(next_pos[1] - self.goal[1])
            
            # Small positive score with goal bias
            best_score = 0.05
            if goal_dist_next < goal_dist_current:
                best_score += 0.1
        
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
            
            # No exploration bonus or penalty - pure geDIG evaluation only
        
        if not action_scores:
            return 'up'
        
        # Softmax selection
        actions = list(action_scores.keys())
        values = np.array(list(action_scores.values()))
        
        temperature = 0.3
        exp_values = np.exp((values - values.max()) / temperature)
        probs = exp_values / exp_values.sum()
        
        return np.random.choice(actions, p=probs)
    
    def navigate(self, max_steps: int = 5000) -> Dict:
        """Navigate maze using donut-based geDIG"""
        print(f"\nDonut-based geDIG Navigation")
        print(f"Start: {self.position}, Goal: {self.goal}")
        print(f"Maze size: {self.width}×{self.height}")
        print(f"Donut search: inner_radius={self.inner_radius}, outer_radius={self.outer_radius}")
        
        steps = 0
        start_time = time.time()
        
        while self.position != self.goal and steps < max_steps:
            if steps % 200 == 0 and steps > 0:
                dist = abs(self.position[0] - self.goal[0]) + abs(self.position[1] - self.goal[1])
                coverage = len(self.visited) / (self.width * self.height) * 100
                
                # Graph stats
                total_edges = sum(len(neighbors) for neighbors in self.graph.values()) // 2
                avg_search_time = np.mean(self.search_times[-100:]) * 1000 if self.search_times else 0
                
                donut_status = "ON" if len(self.episodes) > 100 else "OFF"
                print(f"Step {steps}: pos={self.position}, dist={dist}, "
                      f"coverage={coverage:.1f}%, episodes={len(self.episodes)}, "
                      f"edges={total_edges}, search_time={avg_search_time:.1f}ms, donut={donut_status}")
                
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
        avg_search_time = np.mean(self.search_times) * 1000 if self.search_times else 0
        
        print(f"\nComplete! Success: {success}")
        print(f"Steps: {steps}, Wall hits: {self.wall_hits}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Total episodes: {len(self.episodes)}, Total edges: {total_edges}")
        print(f"Average degree: {total_edges * 2 / len(self.episodes):.1f}")
        print(f"Average search time: {avg_search_time:.2f}ms")
        
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
            'total_edges': total_edges,
            'avg_search_time': avg_search_time
        }


def test_progressive_sizes():
    """Test on progressively larger mazes"""
    from pure_episodic_navigator import create_complex_maze, visualize_maze_with_path
    import os
    
    print("="*70)
    print("VECTOR-BASED geDIG NAVIGATION - PROGRESSIVE TEST")
    print("="*70)
    
    sizes = [15, 20, 25, 30, 35, 40, 45, 50]
    results = {}
    
    for size in sizes:
        print(f"\n{'='*50}")
        print(f"Testing {size}×{size} maze")
        print('='*50)
        
        maze = create_complex_maze(size, seed=42)
        
        # Adaptive parameters
        if size <= 25:
            inner_radius = 0.05
            outer_radius = 0.4
        elif size <= 35:
            inner_radius = 0.1
            outer_radius = 0.5
        else:
            inner_radius = 0.15
            outer_radius = 0.6
        
        nav = DonutGeDIGNavigator(maze, inner_radius=inner_radius, outer_radius=outer_radius)
        
        max_steps = min(size * size * 5, 10000)
        result = nav.navigate(max_steps=max_steps)
        results[size] = result
        
        if result['success']:
            efficiency = result['steps'] / (2 * (size - 2))
            print(f"\n✓ SUCCESS!")
            print(f"Efficiency: {efficiency:.2f}x optimal")
            
            # Save visualization
            os.makedirs('visualizations', exist_ok=True)
            visualize_maze_with_path(
                maze, nav.path,
                f'visualizations/donut_gedig_{size}x{size}.png'
            )
        else:
            print(f"\n✗ Failed")
            print(f"Explored {len(nav.visited)} cells ({len(nav.visited)/(size*size)*100:.1f}%)")
            # Continue to next size even if failed
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n{'Size':<10} {'Success':<10} {'Steps':<10} {'Efficiency':<12} {'Time (s)':<10} {'Search (ms)':<12}")
    print("-" * 80)
    
    for size, result in results.items():
        success = "✓" if result['success'] else "✗"
        steps = result['steps']
        efficiency = f"{steps/(2*(size-2)):.2f}x" if result['success'] else "N/A"
        time_taken = f"{result['time']:.1f}"
        search_time = f"{result['avg_search_time']:.2f}"
        
        print(f"{size}×{size:<5} {success:<10} {steps:<10} {efficiency:<12} {time_taken:<10} {search_time:<12}")


if __name__ == "__main__":
    test_progressive_sizes()