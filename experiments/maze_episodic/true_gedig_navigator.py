#!/usr/bin/env python3
"""
True geDIG Navigator with Graph Persistence
===========================================

Implements proper geDIG evaluation with persistent graph structure.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import time
import json
from collections import defaultdict
import sqlite3
import os

class TrueGeDIGNavigator:
    """Navigator with true geDIG evaluation and graph persistence"""
    
    def __init__(self, maze: np.ndarray, db_path: str = "maze_graph.db"):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.goal = (self.width-2, self.height-2)
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        
        # Database for graph persistence
        self.db_path = db_path
        self._init_database()
        
        # Episode storage
        self.episodes = []
        self.episode_count = 0
        
        # Visual memory
        self.visual_memory = {}
        self._update_visual_memory(1, 1)
        
        # Statistics
        self.wall_hits = 0
        self.moves = 0
        self.hop_selections = {'1-hop': 0, '2-hop': 0, '3-hop': 0}
        
    def _init_database(self):
        """Initialize database for graph storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Episodes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY,
                pos_x INTEGER,
                pos_y INTEGER,
                action TEXT,
                result TEXT,
                reached_goal INTEGER,
                embedding TEXT
            )
        ''')
        
        # Graph edges table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graph_edges (
                source_id INTEGER,
                target_id INTEGER,
                weight REAL,
                PRIMARY KEY (source_id, target_id),
                FOREIGN KEY (source_id) REFERENCES episodes(id),
                FOREIGN KEY (target_id) REFERENCES episodes(id)
            )
        ''')
        
        # Index for efficient queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_edges_source 
            ON graph_edges(source_id)
        ''')
        
        conn.commit()
        conn.close()
        
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
        """Add episode and update graph"""
        embedding = self.create_episode_embedding(pos, action, result, reached_goal)
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO episodes (pos_x, pos_y, action, result, reached_goal, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (pos[0], pos[1], action, result, int(reached_goal), 
              json.dumps(embedding.tolist())))
        
        episode_id = cursor.lastrowid
        
        # Calculate edges to existing episodes
        cursor.execute('''
            SELECT id, pos_x, pos_y, embedding 
            FROM episodes 
            WHERE id != ?
        ''', (episode_id,))
        
        edges_to_add = []
        for row in cursor.fetchall():
            other_id, other_x, other_y, other_emb_str = row
            
            # Spatial distance
            spatial_dist = abs(pos[0] - other_x) + abs(pos[1] - other_y)
            
            # Only connect nearby episodes
            if spatial_dist <= 3:
                # Embedding similarity
                other_emb = np.array(json.loads(other_emb_str))
                emb_similarity = np.dot(embedding, other_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(other_emb) + 1e-8
                )
                
                # Edge weight combines spatial and embedding similarity
                weight = emb_similarity * np.exp(-spatial_dist * 0.3)
                
                if weight > 0.1:  # Threshold for edge creation
                    edges_to_add.append((episode_id, other_id, weight))
                    edges_to_add.append((other_id, episode_id, weight))  # Bidirectional
        
        # Insert edges
        cursor.executemany('''
            INSERT OR REPLACE INTO graph_edges (source_id, target_id, weight)
            VALUES (?, ?, ?)
        ''', edges_to_add)
        
        conn.commit()
        conn.close()
        
        # Keep in-memory reference
        self.episodes.append({
            'id': episode_id,
            'pos': pos,
            'action': action,
            'result': result,
            'reached_goal': reached_goal,
            'embedding': embedding
        })
        self.episode_count += 1
    
    def get_n_hop_neighbors(self, start_pos: Tuple[int, int], n_hops: int) -> List[Dict]:
        """Get n-hop neighbors using persistent graph"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find closest episode to start position
        cursor.execute('''
            SELECT id, pos_x, pos_y, embedding, reached_goal,
                   ABS(pos_x - ?) + ABS(pos_y - ?) as dist
            FROM episodes
            ORDER BY dist
            LIMIT 1
        ''', (start_pos[0], start_pos[1]))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            return []
        
        start_id = row[0]
        
        # BFS through graph for n hops
        visited = {start_id}
        current_layer = {start_id}
        all_episodes = []
        
        for hop in range(n_hops):
            if not current_layer:
                break
                
            # Get next layer
            placeholders = ','.join('?' * len(current_layer))
            cursor.execute(f'''
                SELECT DISTINCT e.id, e.pos_x, e.pos_y, e.action, 
                       e.result, e.reached_goal, e.embedding, g.weight
                FROM graph_edges g
                JOIN episodes e ON g.target_id = e.id
                WHERE g.source_id IN ({placeholders})
                AND e.id NOT IN ({','.join('?' * len(visited))})
            ''', list(current_layer) + list(visited))
            
            next_layer = set()
            for row in cursor.fetchall():
                ep_id, x, y, action, result, goal, emb_str, weight = row
                next_layer.add(ep_id)
                
                all_episodes.append({
                    'id': ep_id,
                    'pos': (x, y),
                    'action': action,
                    'result': result,
                    'reached_goal': bool(goal),
                    'embedding': np.array(json.loads(emb_str)),
                    'weight': weight,
                    'hop': hop + 1
                })
            
            visited.update(next_layer)
            current_layer = next_layer
        
        conn.close()
        return all_episodes
    
    def gedig_message_pass(self, episodes: List[Dict]) -> Dict[int, np.ndarray]:
        """Message passing using graph structure"""
        if not episodes:
            return {}
        
        # Build local adjacency from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        ep_ids = [ep['id'] for ep in episodes]
        embeddings = {ep['id']: ep['embedding'].copy() for ep in episodes}
        
        # Get edges between these episodes
        placeholders = ','.join('?' * len(ep_ids))
        cursor.execute(f'''
            SELECT source_id, target_id, weight
            FROM graph_edges
            WHERE source_id IN ({placeholders})
            AND target_id IN ({placeholders})
        ''', ep_ids + ep_ids)
        
        # Build adjacency
        adjacency = defaultdict(list)
        for source, target, weight in cursor.fetchall():
            adjacency[source].append((target, weight))
        
        conn.close()
        
        # Message passing rounds
        for _ in range(3):
            new_embeddings = {}
            
            for ep_id in embeddings:
                if ep_id not in adjacency:
                    new_embeddings[ep_id] = embeddings[ep_id]
                    continue
                
                # Weighted message aggregation
                messages = []
                weights = []
                
                for neighbor_id, weight in adjacency[ep_id]:
                    if neighbor_id in embeddings:
                        messages.append(embeddings[neighbor_id])
                        weights.append(weight)
                
                if messages:
                    weights = np.array(weights)
                    weights = weights / weights.sum()
                    
                    # Weighted average
                    avg_message = np.sum([m * w for m, w in zip(messages, weights)], axis=0)
                    
                    # Mix with self
                    new_emb = embeddings[ep_id].copy()
                    new_emb[5] = 0.3 * embeddings[ep_id][5] + 0.7 * avg_message[5]  # Goal
                    new_emb[:5] = 0.7 * embeddings[ep_id][:5] + 0.3 * avg_message[:5]
                    
                    new_embeddings[ep_id] = new_emb
                else:
                    new_embeddings[ep_id] = embeddings[ep_id]
            
            embeddings = new_embeddings
        
        return embeddings
    
    def evaluate_action(self, pos: Tuple[int, int], action: str) -> Tuple[float, int]:
        """Evaluate action using true geDIG with multi-hop"""
        dx, dy = {'up': (0, -1), 'right': (1, 0), 
                 'down': (0, 1), 'left': (-1, 0)}[action]
        next_pos = (pos[0] + dx, pos[1] + dy)
        
        best_score = -float('inf')
        best_hop = 1
        
        for n_hops in [1, 2, 3]:
            episodes = self.get_n_hop_neighbors(next_pos, n_hops)
            
            if not episodes:
                continue
            
            # Run message passing
            updated = self.gedig_message_pass(episodes)
            
            # Calculate score
            score = 0.0
            total_weight = 0.0
            
            for ep in episodes:
                if ep['id'] in updated:
                    # Goal signal and success signal
                    goal_signal = updated[ep['id']][5]
                    success_signal = updated[ep['id']][3]
                    
                    # Use graph weight and hop distance
                    graph_weight = ep.get('weight', 1.0)
                    hop_penalty = 0.8 ** (ep.get('hop', 1) - 1)
                    
                    # Combined score
                    ep_score = (goal_signal * 0.6 + success_signal * 0.4) * graph_weight * hop_penalty
                    score += ep_score
                    total_weight += graph_weight
            
            if total_weight > 0:
                score /= total_weight
            
            # Exploration bonus
            score += n_hops * 0.02
            
            if score > best_score:
                best_score = score
                best_hop = n_hops
        
        self.hop_selections[f'{best_hop}-hop'] += 1
        return best_score, best_hop
    
    def decide_action(self) -> str:
        """Decide action using true geDIG evaluation"""
        visual = self.visual_memory.get(self.position, {})
        
        action_scores = {}
        
        for action in ['up', 'right', 'down', 'left']:
            if visual.get(action) == 'wall':
                action_scores[action] = -10.0
                continue
            
            score, _ = self.evaluate_action(self.position, action)
            action_scores[action] = score
            
            # Exploration bonus
            dx, dy = {'up': (0, -1), 'right': (1, 0), 
                     'down': (0, 1), 'left': (-1, 0)}[action]
            next_pos = (self.position[0] + dx, self.position[1] + dy)
            
            if next_pos not in self.visited:
                action_scores[action] += 2.0
        
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
        """Navigate maze using true geDIG"""
        print(f"\nTrue geDIG Navigation with Graph Persistence")
        print(f"Start: {self.position}, Goal: {self.goal}")
        print(f"Maze size: {self.width}×{self.height}")
        
        steps = 0
        start_time = time.time()
        
        while self.position != self.goal and steps < max_steps:
            if steps % 500 == 0 and steps > 0:
                dist = abs(self.position[0] - self.goal[0]) + abs(self.position[1] - self.goal[1])
                coverage = len(self.visited) / (self.width * self.height) * 100
                
                # Get graph stats
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM graph_edges')
                edge_count = cursor.fetchone()[0]
                conn.close()
                
                print(f"Step {steps}: pos={self.position}, dist={dist}, "
                      f"coverage={coverage:.1f}%, episodes={self.episode_count}, "
                      f"edges={edge_count}")
                
                # Hop distribution
                total = sum(self.hop_selections.values())
                if total > 0:
                    print("  Hop usage:", end=" ")
                    for hop, count in self.hop_selections.items():
                        print(f"{hop}: {count/total*100:.1f}%", end=" ")
                    print()
            
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
                self.moves += 1
                self._update_visual_memory(new_pos[0], new_pos[1])
                
                if new_pos == self.goal:
                    reached_goal = True
            else:
                self.wall_hits += 1
            
            # Add episode with graph update
            self.add_episode(old_pos, action, result, reached_goal)
            steps += 1
        
        elapsed = time.time() - start_time
        success = self.position == self.goal
        
        print(f"\nComplete! Success: {success}")
        print(f"Steps: {steps}, Wall hits: {self.wall_hits}")
        print(f"Time: {elapsed:.2f}s")
        
        # Final stats
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM episodes')
        total_episodes = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM graph_edges')
        total_edges = cursor.fetchone()[0]
        conn.close()
        
        print(f"Total episodes: {total_episodes}, Total edges: {total_edges}")
        print(f"Average degree: {total_edges / (total_episodes + 1):.1f}")
        
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
            'total_episodes': total_episodes,
            'total_edges': total_edges
        }


def test_true_gedig():
    """Test true geDIG navigator"""
    from pure_episodic_navigator import create_complex_maze, visualize_maze_with_path
    
    print("="*70)
    print("TRUE geDIG NAVIGATION TEST")
    print("="*70)
    
    # Test on 50x50
    size = 50
    print(f"\n{'='*50}")
    print(f"Testing {size}×{size} maze")
    print('='*50)
    
    # Clean database
    db_path = f"maze_graph_{size}x{size}.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    maze = create_complex_maze(size, seed=42)
    nav = TrueGeDIGNavigator(maze, db_path=db_path)
    
    result = nav.navigate(max_steps=size * size * 5)
    
    if result['success']:
        efficiency = result['steps'] / (2 * (size - 2))
        print(f"✓ Efficiency: {efficiency:.2f}x optimal")
        
        # Save visualization
        os.makedirs('visualizations', exist_ok=True)
        visualize_maze_with_path(
            maze, nav.path,
            f'visualizations/true_gedig_{size}x{size}.png'
        )
        
        print(f"\nGraph density: {result['total_edges'] / result['total_episodes']:.1f} edges per episode")
    else:
        print("✗ Failed")
        print(f"Explored {len(nav.visited)} cells")


if __name__ == "__main__":
    test_true_gedig()