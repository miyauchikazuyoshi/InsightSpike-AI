#!/usr/bin/env python3
"""
Pure Episodic with True geDIG Flow
==================================

正しいgeDIGフローに基づく実装：
1. クエリ生成
2. ドーナツ検索
3. geDIG最小化
4. 洞察エピソード形成
5. 方向正規化
"""

import numpy as np
from typing import Dict, List, Tuple, Set
import networkx as nx
import time

class PureEpisodicTrueFlow:
    """Pure episodic navigation with correct geDIG flow"""
    
    def __init__(self, maze: np.ndarray, 
                 outer_radius: float = 1.5,
                 inner_radius: float = 0.0):
        self.maze = maze
        self.height, self.width = maze.shape
        self.start_pos = (1, 1)
        self.goal_pos = (self.width-2, self.height-2)
        self.position = self.start_pos
        
        # Parameters
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        
        # Episode memory
        self.episodes = []
        self.episode_graph = nx.Graph()
        
        # Visit tracking (7th dimension)
        self.visit_counts = {}
        
        # Statistics
        self.path = [self.position]
        self.steps = 0
        
        # Initialize
        self._create_initial_episodes()
    
    def _create_initial_episodes(self):
        """Create initial episodes from start position"""
        # Try all 4 directions from start
        for action, (dx, dy) in [('up', (0, -1)), ('down', (0, 1)), 
                                  ('left', (-1, 0)), ('right', (1, 0))]:
            next_pos = (self.start_pos[0] + dx, self.start_pos[1] + dy)
            
            if (0 <= next_pos[0] < self.width and 
                0 <= next_pos[1] < self.height and
                self.maze[next_pos[1], next_pos[0]] == 0):
                self.add_episode(self.start_pos, action, 'success', False)
            else:
                self.add_episode(self.start_pos, action, 'wall', False)
        
        # Goal episode
        self.add_episode(self.goal_pos, 'up', 'success', True)
    
    def create_episode_vector(self, pos: Tuple[int, int], action: str, 
                             result: str, reached_goal: bool, visit_count: int) -> np.ndarray:
        """Create 7D episode vector"""
        direction_encoding = {
            'up': 0.0, 'right': 0.25, 'down': 0.5, 'left': 0.75
        }
        
        result_encoding = {
            'success': 1.0, 'wall': 0.0
        }
        
        # Check if position is wall or path
        is_wall = self.maze[pos[1], pos[0]] == 1
        
        vector = np.array([
            pos[0] / self.width,                    # x座標
            pos[1] / self.height,                   # y座標
            direction_encoding[action],             # 移動方向
            result_encoding[result],                # 結果
            0.0 if is_wall else 1.0,               # 壁/通路
            1.0 if reached_goal else 0.0,          # ゴール
            visit_count / 10.0                      # 訪問回数
        ])
        
        return vector / np.linalg.norm(vector)
    
    def add_episode(self, pos: Tuple[int, int], action: str, 
                   result: str, reached_goal: bool):
        """Add episode and update graph"""
        # Update visit count
        pos_key = f"{pos[0]},{pos[1]}"
        self.visit_counts[pos_key] = self.visit_counts.get(pos_key, 0) + 1
        
        # Create vector
        vector = self.create_episode_vector(
            pos, action, result, reached_goal, 
            self.visit_counts[pos_key]
        )
        
        episode = {
            'id': len(self.episodes),
            'pos': pos,
            'action': action,
            'result': result,
            'vector': vector,
            'reached_goal': reached_goal
        }
        
        # Add to graph
        self.episode_graph.add_node(episode['id'])
        
        # Find similar episodes and create edges
        for i, other_ep in enumerate(self.episodes):
            cos_sim = np.dot(vector, other_ep['vector'])
            if cos_sim > 0.7:
                self.episode_graph.add_edge(episode['id'], i, weight=cos_sim)
        
        self.episodes.append(episode)
    
    def create_query(self, position: Tuple[int, int]) -> np.ndarray:
        """Create query vector for current position"""
        pos_key = f"{position[0]},{position[1]}"
        visit_count = self.visit_counts.get(pos_key, 0)
        
        query = np.array([
            position[0] / self.width,   # 現x座標
            position[1] / self.height,  # 現y座標
            0.5,                        # null (方向)
            0.5,                        # null (結果)
            0.5,                        # null (壁/通路)
            1.0,                        # ゴール（目的）
            visit_count / 10.0          # 訪問回数
        ])
        
        return query / np.linalg.norm(query)
    
    def donut_search(self, query: np.ndarray) -> List[int]:
        """Donut search for similar episodes"""
        candidates = []
        
        for i, ep in enumerate(self.episodes):
            distance = np.linalg.norm(query - ep['vector'])
            if distance <= self.outer_radius:  # inner_radius = 0
                candidates.append(i)
        
        return candidates
    
    def calculate_gedig(self, episode_ids: List[int]) -> float:
        """Calculate geDIG for episode subset"""
        if not episode_ids:
            return float('inf')
        
        # Graph edit distance
        subgraph = self.episode_graph.subgraph(episode_ids)
        ged = len(episode_ids) + subgraph.number_of_edges()
        
        # Information gain
        ig = 0.0
        for ep_id in episode_ids:
            ig += self.episodes[ep_id]['vector'][5]  # Goal component
        
        ig = ig / len(episode_ids) if episode_ids else 0.0
        
        return ged - ig
    
    def find_minimum_gedig_subset(self, candidate_ids: List[int]) -> List[int]:
        """Find subset that minimizes geDIG"""
        if not candidate_ids:
            return []
        
        best_subset = []
        best_gedig = float('inf')
        
        # Try subsets (simplified - try singles and pairs)
        for i, id1 in enumerate(candidate_ids):
            # Single episode
            gedig = self.calculate_gedig([id1])
            if gedig < best_gedig:
                best_gedig = gedig
                best_subset = [id1]
            
            # Pairs
            for id2 in candidate_ids[i+1:]:
                gedig = self.calculate_gedig([id1, id2])
                if gedig < best_gedig:
                    best_gedig = gedig
                    best_subset = [id1, id2]
        
        return best_subset
    
    def form_insight_episode(self, episode_ids: List[int]) -> np.ndarray:
        """Form insight episode from selected episodes"""
        if not episode_ids:
            return np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0])
        
        # Get vectors
        vectors = [self.episodes[ep_id]['vector'] for ep_id in episode_ids]
        
        # Weight by goal component
        weights = np.array([v[5] + 0.1 for v in vectors])
        weights = weights / weights.sum()
        
        # Weighted average
        insight = np.zeros(7)
        for i, vector in enumerate(vectors):
            insight += weights[i] * vector
        
        return insight
    
    def normalize_direction(self, insight_vector: np.ndarray) -> str:
        """Get direction from insight vector"""
        direction_value = insight_vector[2]
        
        if direction_value < 0.125:
            return 'up'
        elif direction_value < 0.375:
            return 'right'
        elif direction_value < 0.625:
            return 'down'
        else:
            return 'left'
    
    def navigate_step(self) -> bool:
        """One navigation step following true geDIG flow"""
        # 1. Create query
        query = self.create_query(self.position)
        
        # 2. Donut search
        candidates = self.donut_search(query)
        
        # 3. Find minimum geDIG subset
        selected = self.find_minimum_gedig_subset(candidates)
        
        # 4. Form insight episode
        insight = self.form_insight_episode(selected)
        
        # 5. Normalize to direction
        direction = self.normalize_direction(insight)
        
        # Execute action
        dx, dy = {'up': (0, -1), 'right': (1, 0), 
                 'down': (0, 1), 'left': (-1, 0)}[direction]
        new_pos = (self.position[0] + dx, self.position[1] + dy)
        
        # Check result
        result = 'wall'
        if (0 <= new_pos[0] < self.width and 
            0 <= new_pos[1] < self.height and
            self.maze[new_pos[1], new_pos[0]] == 0):
            result = 'success'
            self.position = new_pos
            self.path.append(new_pos)
        
        reached_goal = (self.position == self.goal_pos)
        
        # Add episode
        self.add_episode(self.position, direction, result, reached_goal)
        
        self.steps += 1
        
        return reached_goal
    
    def navigate(self, max_steps: int = 1000) -> Dict:
        """Complete navigation"""
        print(f"Pure Episodic with True geDIG Flow")
        print(f"Maze size: {self.width}x{self.height}")
        print(f"Start: {self.start_pos}, Goal: {self.goal_pos}")
        
        start_time = time.time()
        
        while self.steps < max_steps:
            if self.steps % 50 == 0:
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
            'graph_edges': self.episode_graph.number_of_edges(),
            'time': elapsed
        }


def test_true_flow():
    """Test with true geDIG flow"""
    from pure_episodic_navigator import create_complex_maze, visualize_maze_with_path
    
    # Test different sizes
    for size in [5, 10]:
        print(f"\n{'='*60}")
        print(f"Testing {size}x{size} maze")
        print(f"{'='*60}\n")
        
        maze = create_complex_maze(size, seed=42)
        nav = PureEpisodicTrueFlow(maze)
        
        result = nav.navigate(max_steps=size*size*3)
        
        print(f"\nResults:")
        print(f"- Success: {result['success']}")
        print(f"- Steps: {result['steps']}")
        print(f"- Episodes: {result['episodes']}")
        print(f"- Graph edges: {result['graph_edges']}")
        print(f"- Time: {result['time']:.2f}s")
        
        if result['success']:
            filename = f'pure_true_flow_{size}x{size}.png'
            visualize_maze_with_path(maze, result['path'], filename)
            print(f"- Saved: {filename}")


if __name__ == "__main__":
    test_true_flow()