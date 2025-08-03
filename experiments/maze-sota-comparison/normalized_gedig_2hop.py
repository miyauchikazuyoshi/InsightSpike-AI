#!/usr/bin/env python3
"""
Normalized geDIG 2-hop Navigator
================================

Based on the reference implementation with proper normalization
and similarity-based decision making.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math
import time
from datetime import datetime


@dataclass
class NormalizedEpisode:
    """正規化されたエピソード"""
    episode_type: str  # "goal", "movement", "visual"
    content: Dict
    raw_vector: np.ndarray
    normalized_vector: np.ndarray
    episode_id: int
    timestamp: int
    
    # geDIG related
    ged_delta: float = 0.0
    ig_delta: float = 0.0
    gedig_value: float = 0.0


class NormalizedGeDIG2HopNavigator:
    """Normalized geDIG navigator with proper 2-hop evaluation"""
    
    def __init__(self, maze: np.ndarray, use_2hop: bool = True):
        self.maze = maze
        self.height, self.width = maze.shape
        self.maze_size = max(self.height, self.width)
        self.position = (1, 1)
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        self.use_2hop = use_2hop
        
        # Episode memory
        self.episodes: List[NormalizedEpisode] = []
        self.episode_counter = 0
        self.current_step = 0
        
        # Statistics
        self.position_visit_counts = {(1, 1): 1}
        self.gedig_spikes = []
        
        # Initialize with goal episode
        self._add_goal_episode()
        
    def _add_goal_episode(self):
        """Add normalized goal episode"""
        # Goal with abstract coordinates
        raw_vector = np.array([self.width-2, self.height-2, 100.0])  # High value for goal
        normalized_vector = self._normalize_vector(raw_vector, "goal")
        
        episode = NormalizedEpisode(
            episode_type="goal",
            content={"position": (self.width-2, self.height-2)},
            raw_vector=raw_vector,
            normalized_vector=normalized_vector,
            episode_id=self.episode_counter,
            timestamp=self.current_step
        )
        
        self.episodes.append(episode)
        self.episode_counter += 1
        
    def _normalize_vector(self, raw_vector: np.ndarray, episode_type: str) -> np.ndarray:
        """Normalize vector to [-1, 1] range"""
        if episode_type == "goal":
            normalized = np.zeros(10)
            normalized[0] = (raw_vector[0] / self.maze_size) * 2 - 1  # X
            normalized[1] = (raw_vector[1] / self.maze_size) * 2 - 1  # Y
            normalized[2] = 1.0  # Goal flag
            return normalized
            
        elif episode_type == "movement":
            normalized = np.zeros(10)
            
            # Positions
            normalized[0] = (raw_vector[0] / self.maze_size) * 2 - 1  # from_x
            normalized[1] = (raw_vector[1] / self.maze_size) * 2 - 1  # from_y
            normalized[2] = (raw_vector[2] / self.maze_size) * 2 - 1  # to_x
            normalized[3] = (raw_vector[3] / self.maze_size) * 2 - 1  # to_y
            
            # Result
            normalized[4] = raw_vector[4]  # -1 or 1
            
            # Action
            normalized[5] = -1.0 + (raw_vector[5] / 1.5)  # [0,3] → [-1,1]
            
            # Visit counts (log normalized with tanh)
            normalized[6] = np.tanh(raw_vector[6] / 3.0)
            normalized[7] = np.tanh(raw_vector[7] / 3.0)
            
            # Distance
            distance = abs(raw_vector[2] - raw_vector[0]) + abs(raw_vector[3] - raw_vector[1])
            normalized[8] = np.tanh(distance / 5.0)
            
            # Temporal feature
            normalized[9] = np.tanh(self.episode_counter / 1000.0)
            
            return normalized
            
        elif episode_type == "visual":
            # Visual information as movement with result=0
            normalized = np.zeros(10)
            normalized[0] = (raw_vector[0] / self.maze_size) * 2 - 1  # current_x
            normalized[1] = (raw_vector[1] / self.maze_size) * 2 - 1  # current_y
            normalized[2] = (raw_vector[2] / self.maze_size) * 2 - 1  # seen_x
            normalized[3] = (raw_vector[3] / self.maze_size) * 2 - 1  # seen_y
            normalized[4] = raw_vector[4]  # wall=-1, path=1
            normalized[5] = -1.0 + (raw_vector[5] / 1.5)  # direction
            return normalized
            
    def _calculate_gedig(self, episode: NormalizedEpisode, 
                        recent_episodes: List[NormalizedEpisode]) -> Tuple[float, float, float]:
        """Calculate geDIG values using normalized vectors"""
        if not recent_episodes:
            return 1.0, 1.0, 1.0
            
        # Find most similar episode
        max_similarity = 0.0
        for past_ep in recent_episodes:
            # Cosine similarity
            similarity = np.dot(episode.normalized_vector, past_ep.normalized_vector) / (
                np.linalg.norm(episode.normalized_vector) * np.linalg.norm(past_ep.normalized_vector) + 1e-10
            )
            max_similarity = max(max_similarity, similarity)
            
        # GED delta (novelty)
        ged_delta = 1.0 - max_similarity
        
        # IG delta (information gain)
        if episode.episode_type == "movement":
            from_pos = tuple(episode.content['from'])
            visit_count = self.position_visit_counts.get(from_pos, 0)
            ig_delta = 1.0 / (1.0 + math.log(1 + visit_count))
        else:
            ig_delta = 1.0
            
        # geDIG value
        gedig_value = ged_delta * ig_delta
        
        # Spike detection
        if gedig_value < -0.5:  # Threshold for spike
            self.gedig_spikes.append({
                'step': self.current_step,
                'position': self.position,
                'value': gedig_value
            })
            
        return ged_delta, ig_delta, gedig_value
        
    def _record_visual_information(self, x: int, y: int):
        """Record what we see from current position"""
        directions = {
            'up': (0, 0),
            'right': (1, 1), 
            'down': (2, 2),
            'left': (3, 3)
        }
        
        for direction, (action, dir_idx) in directions.items():
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < self.width and 0 <= ny < self.height:
                wall_or_path = 1.0 if self.maze[ny, nx] == 0 else -1.0
                visit_count = self.position_visit_counts.get((nx, ny), 0)
                
                raw_vector = np.array([
                    x, y, nx, ny, wall_or_path, action,
                    math.log(1 + self.position_visit_counts.get((x, y), 0)),
                    math.log(1 + visit_count)
                ])
                
                normalized_vector = self._normalize_vector(raw_vector, "visual")
                
                episode = NormalizedEpisode(
                    episode_type="visual",
                    content={
                        "from": (x, y),
                        "to": (nx, ny),
                        "direction": direction,
                        "wall_or_path": wall_or_path
                    },
                    raw_vector=raw_vector,
                    normalized_vector=normalized_vector,
                    episode_id=self.episode_counter,
                    timestamp=self.current_step
                )
                
                # Calculate geDIG
                recent = self.episodes[-50:] if len(self.episodes) > 50 else self.episodes
                ged, ig, gedig = self._calculate_gedig(episode, recent)
                episode.ged_delta = ged
                episode.ig_delta = ig
                episode.gedig_value = gedig
                
                self.episodes.append(episode)
                self.episode_counter += 1
                
    def decide_action(self) -> str:
        """Decide action using similarity-based evaluation"""
        current_pos = self.position
        possible_actions = []
        
        # Check which actions are possible
        for action, (dx, dy) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
            nx, ny = current_pos[0] + dx, current_pos[1] + dy
            if (0 <= nx < self.width and 
                0 <= ny < self.height and 
                self.maze[ny, nx] == 0 and
                (nx, ny) not in self.visited):
                possible_actions.append(action)
                
        if not possible_actions:
            return 'backtrack'
            
        # Evaluate each action
        action_values = {}
        
        for action in possible_actions:
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            # Create hypothetical episode vector
            visit_count = self.position_visit_counts.get(current_pos, 0)
            next_visit_count = self.position_visit_counts.get(next_pos, 0)
            
            test_vector = self._normalize_vector(np.array([
                current_pos[0], current_pos[1],
                next_pos[0], next_pos[1],
                1.0,  # Assume success
                action,
                math.log(1 + visit_count),
                math.log(1 + next_visit_count)
            ]), "movement")
            
            # Calculate expected value based on similarity to past episodes
            if self.episodes:
                similarities = []
                values = []
                
                # Consider recent episodes
                for ep in self.episodes[-30:]:
                    if ep.episode_type in ["movement", "visual"]:
                        sim = np.dot(test_vector, ep.normalized_vector) / (
                            np.linalg.norm(test_vector) * np.linalg.norm(ep.normalized_vector) + 1e-10
                        )
                        similarities.append(sim)
                        
                        # Value based on episode outcome
                        if ep.episode_type == "movement":
                            values.append(ep.gedig_value)
                        else:
                            # Visual episodes
                            values.append(0.5 if ep.content.get('wall_or_path', 1) > 0 else -0.5)
                            
                if similarities:
                    # Weighted average
                    weights = np.array(similarities)
                    weights = np.exp(weights * 5)  # Temperature parameter
                    weights /= (weights.sum() + 1e-10)
                    expected_value = np.dot(weights, values)
                else:
                    expected_value = 1.0
            else:
                expected_value = 1.0
                
            # Exploration bonus
            if next_visit_count == 0:
                expected_value *= 2.0
                
            # 2-hop evaluation
            if self.use_2hop and len(self.episodes) > 100:
                # Check if this action leads to better connectivity
                two_hop_bonus = self._evaluate_2hop_connectivity(next_pos)
                expected_value += two_hop_bonus
                
            action_values[action] = expected_value
            
        # Softmax selection
        if action_values:
            actions = list(action_values.keys())
            values = np.array(list(action_values.values()))
            
            # Temperature for exploration
            temperature = 0.2
            exp_values = np.exp(values / temperature)
            probs = exp_values / exp_values.sum()
            
            chosen_action = np.random.choice(actions, p=probs)
            return ['up', 'right', 'down', 'left'][chosen_action]
            
        return 'backtrack'
        
    def _evaluate_2hop_connectivity(self, next_pos: Tuple[int, int]) -> float:
        """Evaluate 2-hop connectivity bonus"""
        bonus = 0.0
        
        # Check how many new positions become reachable
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nnx, nny = next_pos[0] + dx, next_pos[1] + dy
            if (0 <= nnx < self.width and 
                0 <= nny < self.height and 
                self.maze[nny, nnx] == 0 and
                (nnx, nny) not in self.visited):
                bonus += 0.1
                
        return bonus
        
    def navigate(self, goal: Tuple[int, int], max_steps: int = 3000) -> Dict:
        """Navigate maze using normalized geDIG"""
        
        print(f"Normalized geDIG {'2-hop' if self.use_2hop else '1-hop'} Navigation")
        print(f"Start: {self.position}, Goal: {goal}")
        
        steps = 0
        backtrack_count = 0
        
        # Record initial visual information
        self._record_visual_information(1, 1)
        
        start_time = time.time()
        
        while self.position != goal and steps < max_steps:
            # Decide action
            action = self.decide_action()
            
            # Progress report
            if steps % 100 == 0:
                dist = abs(self.position[0] - goal[0]) + abs(self.position[1] - goal[1])
                print(f"Step {steps}: pos={self.position}, dist={dist}, "
                      f"episodes={len(self.episodes)}, spikes={len(self.gedig_spikes)}")
                
            # Execute action
            if action == 'backtrack':
                backtrack_count += 1
                if len(self.path) > 1:
                    self.path.pop()
                    self.position = self.path[-1]
            else:
                # Get next position
                action_idx = ['up', 'right', 'down', 'left'].index(action)
                dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action_idx]
                next_pos = (self.position[0] + dx, self.position[1] + dy)
                
                # Record movement episode
                from_visits = self.position_visit_counts.get(self.position, 0)
                to_visits = self.position_visit_counts.get(next_pos, 0)
                
                raw_vector = np.array([
                    self.position[0], self.position[1],
                    next_pos[0], next_pos[1],
                    1.0,  # Success
                    action_idx,
                    math.log(1 + from_visits),
                    math.log(1 + to_visits)
                ])
                
                normalized_vector = self._normalize_vector(raw_vector, "movement")
                
                episode = NormalizedEpisode(
                    episode_type="movement",
                    content={
                        "from": self.position,
                        "to": next_pos,
                        "action": action_idx,
                        "result": "success"
                    },
                    raw_vector=raw_vector,
                    normalized_vector=normalized_vector,
                    episode_id=self.episode_counter,
                    timestamp=self.current_step
                )
                
                # Calculate geDIG
                recent = self.episodes[-50:] if len(self.episodes) > 50 else self.episodes
                ged, ig, gedig = self._calculate_gedig(episode, recent)
                episode.ged_delta = ged
                episode.ig_delta = ig
                episode.gedig_value = gedig
                
                self.episodes.append(episode)
                self.episode_counter += 1
                
                # Move
                self.position = next_pos
                self.visited.add(next_pos)
                self.path.append(next_pos)
                self.position_visit_counts[next_pos] = to_visits + 1
                
                # Record new visual information
                self._record_visual_information(next_pos[0], next_pos[1])
                
            steps += 1
            self.current_step += 1
            
        elapsed_time = time.time() - start_time
        success = self.position == goal
        
        print(f"\nNavigation complete: success={success}, steps={steps}, "
              f"time={elapsed_time:.2f}s")
        print(f"Total episodes: {len(self.episodes)}, Spikes: {len(self.gedig_spikes)}")
        
        return {
            'success': success,
            'steps': steps,
            'visited_count': len(self.visited),
            'episode_count': len(self.episodes),
            'backtrack_count': backtrack_count,
            'spike_count': len(self.gedig_spikes),
            'elapsed_time': elapsed_time,
            'efficiency': len(self.visited) / steps * 100 if steps > 0 else 0
        }


def test_normalized_gedig():
    """Test normalized geDIG navigation"""
    
    # Generate maze
    np.random.seed(42)
    size = 50  # Larger maze
    maze = np.ones((size, size), dtype=int)
    
    # Recursive backtracker
    def carve_passages(cx, cy):
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        np.random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < size and 0 <= ny < size and maze[ny, nx] == 1:
                maze[cy + dy // 2, cx + dx // 2] = 0
                maze[ny, nx] = 0
                carve_passages(nx, ny)
    
    maze[1, 1] = 0
    carve_passages(1, 1)
    maze[size-2, size-2] = 0
    
    # Add extra paths
    for _ in range(size // 2):
        x = np.random.randint(1, size-1)
        y = np.random.randint(1, size-1)
        if maze[y, x] == 1:
            neighbors = sum(1 for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]
                          if 0 <= x+dx < size and 0 <= y+dy < size and maze[y+dy, x+dx] == 0)
            if neighbors >= 2:
                maze[y, x] = 0
    
    print("="*70)
    print("NORMALIZED geDIG NAVIGATION TEST")
    print("="*70)
    
    goal = (size-2, size-2)
    
    # Test 1-hop
    print("\n--- 1-hop Normalized geDIG ---")
    nav_1hop = NormalizedGeDIG2HopNavigator(maze, use_2hop=False)
    result_1hop = nav_1hop.navigate(goal, max_steps=2000)
    
    # Test 2-hop
    print("\n--- 2-hop Normalized geDIG ---")
    nav_2hop = NormalizedGeDIG2HopNavigator(maze, use_2hop=True)
    result_2hop = nav_2hop.navigate(goal, max_steps=2000)
    
    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    for name, result in [("1-hop", result_1hop), ("2-hop", result_2hop)]:
        print(f"\n{name}:")
        print(f"  Success: {result['success']}")
        print(f"  Steps: {result['steps']}")
        print(f"  Efficiency: {result['efficiency']:.1f}%")
        print(f"  Spikes: {result['spike_count']}")
        print(f"  Episodes: {result['episode_count']}")
    
    if result_1hop['steps'] > 0 and result_2hop['steps'] > 0:
        print(f"\nImprovement with 2-hop:")
        print(f"  Steps: {(1 - result_2hop['steps']/result_1hop['steps'])*100:+.1f}%")
        print(f"  Efficiency: {result_2hop['efficiency'] - result_1hop['efficiency']:+.1f}%")


if __name__ == "__main__":
    test_normalized_gedig()