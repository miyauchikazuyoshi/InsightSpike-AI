#!/usr/bin/env python3
"""
Truly Pure Multi-hop geDIG Navigator
====================================

No if statements for wall checking. All decisions based on 
similarity to past episodes.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
from datetime import datetime
import json

@dataclass
class MemoryEpisode:
    """Episode with features and outcome"""
    position: Tuple[int, int]
    action: str
    visual_features: np.ndarray  # What was seen
    outcome_features: np.ndarray  # What happened
    reward: float  # +1 for successful move, -1 for wall hit
    timestamp: int

class TrulyPureNavigator:
    """Navigator that learns purely from experience"""
    
    def __init__(self, maze: np.ndarray, hop_count: int = 1):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        self.hop_count = hop_count
        
        # Episode memory
        self.episodes: List[MemoryEpisode] = []
        self.position_visits = {(1, 1): 1}
        self.step_count = 0
        
        # Statistics
        self.wall_hits = 0
        self.successful_moves = 0
        self.coverage_over_time = []
        
        # Feature dimensions
        self.visual_feature_dim = 5  # position (2) + visual context (3)
        self.outcome_feature_dim = 3  # next position (2) + success (1)
        
    def _get_visual_features(self, pos: Tuple[int, int], action: str) -> np.ndarray:
        """Get visual features for current state-action pair"""
        x, y = pos
        
        # Normalize position
        norm_x = (x / self.width) * 2 - 1
        norm_y = (y / self.height) * 2 - 1
        
        # Action encoding (one-hot style but continuous)
        action_encoding = {
            'up': -1.0,
            'right': -0.33,
            'down': 0.33,
            'left': 1.0
        }[action]
        
        # Visit count (normalized)
        visit_count = self.position_visits.get(pos, 0)
        norm_visits = np.tanh(visit_count / 5.0)
        
        # Time feature
        time_feature = np.tanh(self.step_count / 1000.0)
        
        features = np.array([
            norm_x,
            norm_y,
            action_encoding,
            norm_visits,
            time_feature
        ])
        
        return features
    
    def _get_outcome_features(self, old_pos: Tuple[int, int], 
                            new_pos: Tuple[int, int], 
                            success: bool) -> np.ndarray:
        """Get outcome features after attempting action"""
        # Normalize new position
        norm_x = (new_pos[0] / self.width) * 2 - 1
        norm_y = (new_pos[1] / self.height) * 2 - 1
        
        # Success indicator
        success_val = 1.0 if success else -1.0
        
        return np.array([norm_x, norm_y, success_val])
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_expected_value(self, visual_features: np.ndarray) -> float:
        """Calculate expected value based on similarity to past episodes"""
        if not self.episodes:
            # No experience yet - neutral value with small random exploration
            return np.random.normal(0.0, 0.1)
        
        # Find similar episodes
        similarities = []
        rewards = []
        
        for episode in self.episodes:
            # Compare visual features
            sim = self._cosine_similarity(visual_features, episode.visual_features)
            
            # Only consider reasonably similar episodes
            if sim > 0.5:
                similarities.append(sim)
                rewards.append(episode.reward)
        
        if not similarities:
            # No similar experience - encourage exploration
            return 0.1
        
        # Weighted average based on similarity
        similarities = np.array(similarities)
        rewards = np.array(rewards)
        
        # Softmax weights
        weights = np.exp(similarities * 5.0)  # Temperature parameter
        weights = weights / weights.sum()
        
        expected_value = np.dot(weights, rewards)
        
        # Add exploration bonus for unvisited states
        exploration_bonus = 0.0
        if len(similarities) < 3:  # Few similar experiences
            exploration_bonus = 0.2
        
        return expected_value + exploration_bonus
    
    def _calculate_n_hop_value(self, action: str, current_pos: Tuple[int, int]) -> float:
        """Calculate value with n-hop lookahead (based on experience)"""
        # Get visual features for this state-action
        visual_features = self._get_visual_features(current_pos, action)
        
        # Base value from direct experience
        base_value = self._calculate_expected_value(visual_features)
        
        # For multi-hop, we need to imagine future states
        if self.hop_count > 1 and self.episodes:
            # Find episodes similar to taking this action
            future_value = 0.0
            
            for episode in self.episodes[-50:]:  # Recent episodes
                sim = self._cosine_similarity(visual_features, episode.visual_features)
                if sim > 0.7 and episode.reward > 0:  # Similar successful move
                    # Imagine being at that next position
                    next_pos = self._extract_position_from_outcome(episode.outcome_features)
                    if next_pos:
                        # Recursively evaluate future positions
                        future_sum = 0.0
                        for future_action in ['up', 'right', 'down', 'left']:
                            future_features = self._get_visual_features(next_pos, future_action)
                            future_sum += self._calculate_expected_value(future_features)
                        future_value += (future_sum / 4.0) * sim * 0.5
            
            base_value += future_value
        
        return base_value
    
    def _extract_position_from_outcome(self, outcome_features: np.ndarray) -> Optional[Tuple[int, int]]:
        """Extract position from outcome features"""
        if outcome_features[2] < 0:  # Failed move
            return None
        
        # Denormalize position
        x = int((outcome_features[0] + 1) * self.width / 2)
        y = int((outcome_features[1] + 1) * self.height / 2)
        
        # Bounds check
        if 0 <= x < self.width and 0 <= y < self.height:
            return (x, y)
        return None
    
    def decide_action(self) -> str:
        """Decide action based purely on past experience"""
        current_pos = self.position
        action_values = {}
        
        # Evaluate all actions equally
        for action in ['up', 'right', 'down', 'left']:
            value = self._calculate_n_hop_value(action, current_pos)
            action_values[action] = value
        
        # Softmax selection
        actions = list(action_values.keys())
        values = np.array(list(action_values.values()))
        
        # Temperature for exploration
        temperature = 0.5
        
        # Softmax probabilities
        exp_values = np.exp(values / temperature)
        probs = exp_values / exp_values.sum()
        
        # Choose action
        chosen_action = np.random.choice(actions, p=probs)
        
        return chosen_action
    
    def execute_action(self, action: str) -> bool:
        """Execute action and learn from outcome"""
        old_pos = self.position
        
        # Get visual features before action
        visual_features = self._get_visual_features(old_pos, action)
        
        # Attempt to move
        dx, dy = {'up': (0, -1), 'right': (1, 0), 
                  'down': (0, 1), 'left': (-1, 0)}[action]
        new_pos = (old_pos[0] + dx, old_pos[1] + dy)
        
        # Check if move is valid (for recording outcome, not for decision)
        success = False
        reward = -1.0  # Default: failure
        
        if (0 <= new_pos[0] < self.width and 
            0 <= new_pos[1] < self.height and 
            self.maze[new_pos[1], new_pos[0]] == 0):
            # Successful move
            success = True
            reward = 1.0
            self.successful_moves += 1
            
            # Update position
            self.position = new_pos
            if new_pos not in self.visited:
                self.visited.add(new_pos)
            self.path.append(new_pos)
            self.position_visits[new_pos] = self.position_visits.get(new_pos, 0) + 1
        else:
            # Hit wall or boundary
            self.wall_hits += 1
        
        # Get outcome features
        outcome_features = self._get_outcome_features(old_pos, new_pos, success)
        
        # Record episode
        episode = MemoryEpisode(
            position=old_pos,
            action=action,
            visual_features=visual_features,
            outcome_features=outcome_features,
            reward=reward,
            timestamp=self.step_count
        )
        self.episodes.append(episode)
        
        return success
    
    def navigate(self, goal: Tuple[int, int], max_steps: int = 10000) -> Dict:
        """Navigate maze learning from experience"""
        
        print(f"\nTruly Pure {self.hop_count}-hop Navigation")
        print(f"Start: {self.position}, Goal: {goal}")
        print("Learning from scratch - no hardcoded wall avoidance!")
        
        steps = 0
        
        # Track walkable cells
        walkable_cells = sum(1 for y in range(self.height) 
                           for x in range(self.width) 
                           if self.maze[y, x] == 0)
        
        start_time = time.time()
        
        while self.position != goal and steps < max_steps:
            # Track coverage
            if steps % 100 == 0:
                coverage = len(self.visited) / walkable_cells * 100
                self.coverage_over_time.append((steps, coverage))
            
            # Progress report
            if steps % 1000 == 0 and steps > 0:
                dist = abs(self.position[0] - goal[0]) + abs(self.position[1] - goal[1])
                coverage = len(self.visited) / walkable_cells * 100
                success_rate = (self.successful_moves / (self.successful_moves + self.wall_hits) * 100 
                               if (self.successful_moves + self.wall_hits) > 0 else 0)
                print(f"Step {steps}: pos={self.position}, dist={dist}, "
                      f"coverage={coverage:.1f}%, wall_hits={self.wall_hits}, "
                      f"success_rate={success_rate:.1f}%")
            
            # Decide and execute action
            action = self.decide_action()
            self.execute_action(action)
            
            steps += 1
            self.step_count += 1
        
        elapsed_time = time.time() - start_time
        success = self.position == goal
        final_coverage = len(self.visited) / walkable_cells * 100
        
        print(f"\nNavigation complete: success={success}, steps={steps}")
        print(f"Coverage: {final_coverage:.1f}%, Wall hits: {self.wall_hits}")
        print(f"Success rate: {self.successful_moves/(self.successful_moves+self.wall_hits)*100:.1f}%")
        print(f"Total episodes learned: {len(self.episodes)}")
        
        return {
            'success': success,
            'steps': steps,
            'visited_count': len(self.visited),
            'coverage': final_coverage,
            'wall_hits': self.wall_hits,
            'successful_moves': self.successful_moves,
            'move_success_rate': self.successful_moves / (self.successful_moves + self.wall_hits) * 100,
            'elapsed_time': elapsed_time,
            'episode_count': len(self.episodes)
        }


def test_truly_pure_implementation(num_runs=5):
    """Test truly pure implementation"""
    
    print("="*70)
    print("TRULY PURE IMPLEMENTATION TEST")
    print("No if statements, no wall checking, pure learning!")
    print("="*70)
    
    all_results = {
        '1-hop': [],
        '2-hop': [],
        '3-hop': []
    }
    
    for run in range(num_runs):
        print(f"\n{'='*30} RUN {run + 1}/{num_runs} {'='*30}")
        
        # Generate maze
        np.random.seed(42 + run)
        size = 50
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
        
        # Add loops
        for _ in range(size):
            x = np.random.randint(2, size-2)
            y = np.random.randint(2, size-2)
            if maze[y, x] == 1:
                neighbors = sum(1 for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]
                              if 0 <= x+dx < size and 0 <= y+dy < size and maze[y+dy, x+dx] == 0)
                if neighbors >= 2:
                    maze[y, x] = 0
        
        goal = (size-2, size-2)
        
        # Test different hop counts
        for hop_count in [1, 2, 3]:
            navigator = TrulyPureNavigator(maze, hop_count=hop_count)
            result = navigator.navigate(goal, max_steps=10000)
            all_results[f'{hop_count}-hop'].append(result)
    
    # Analysis
    print("\n" + "="*70)
    print("TRULY PURE RESULTS SUMMARY")
    print("="*70)
    
    # Calculate statistics
    stats = {}
    for method in all_results:
        method_results = all_results[method]
        successful = [r for r in method_results if r['success']]
        
        stats[method] = {
            'success_rate': len(successful) / num_runs * 100,
            'avg_steps_all': np.mean([r['steps'] for r in method_results]),
            'avg_steps_success': np.mean([r['steps'] for r in successful]) if successful else None,
            'avg_coverage': np.mean([r['coverage'] for r in method_results]),
            'avg_wall_hits': np.mean([r['wall_hits'] for r in method_results]),
            'avg_move_success_rate': np.mean([r['move_success_rate'] for r in method_results])
        }
    
    # Display results
    print("\n| Method | Success% | Avg Steps | Wall Hits | Move Success% | Coverage% |")
    print("|--------|----------|-----------|-----------|---------------|-----------|")
    
    for method, stat in stats.items():
        print(f"| {method:6} | {stat['success_rate']:7.0f}% | {stat['avg_steps_all']:9.0f} | "
              f"{stat['avg_wall_hits']:9.0f} | {stat['avg_move_success_rate']:12.1f}% | "
              f"{stat['avg_coverage']:9.1f} |")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'truly_pure_results_{timestamp}.json', 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'num_runs': num_runs,
            'results': {k: v for k, v in all_results.items()},
            'statistics': stats
        }, f, indent=2)
    
    print(f"\nResults saved to truly_pure_results_{timestamp}.json")
    
    # Compare wall hit patterns over time
    print("\n" + "="*70)
    print("LEARNING CURVE ANALYSIS")
    print("="*70)
    
    # Show how wall hits decrease over time (learning effect)
    for method in ['1-hop', '2-hop', '3-hop']:
        total_wall_hits = sum(r['wall_hits'] for r in all_results[method])
        total_moves = sum(r['wall_hits'] + r['successful_moves'] for r in all_results[method])
        print(f"{method}: {total_wall_hits} wall hits out of {total_moves} total moves")
    
    return all_results, stats


if __name__ == "__main__":
    results, stats = test_truly_pure_implementation(num_runs=5)