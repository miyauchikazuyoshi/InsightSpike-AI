#!/usr/bin/env python3
"""
Test 2-hop evaluation on 50x50 maze with fixed seed
===================================================

Use the same maze generation as null_goal experiment
for fair comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
import time
from datetime import datetime
import json


class FixedMaze2HopNavigator:
    """Maze navigator with 2-hop structural evaluation"""
    
    def __init__(self, maze: np.ndarray):
        self.maze = maze
        self.height, self.width = maze.shape
        self.visited = set()
        self.path = []
        self.decision_history = []
        self.dead_end_cache = set()
        
    def _get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighbors of a position"""
        neighbors = []
        i, j = position
        
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = i + di, j + dj
            if (0 <= ni < self.height and 
                0 <= nj < self.width and 
                self.maze[ni, nj] == 0):
                neighbors.append((ni, nj))
        
        return neighbors
    
    def _count_unvisited_neighbors(self, position: Tuple[int, int]) -> int:
        """Count unvisited neighbors"""
        return sum(1 for n in self._get_neighbors(position) if n not in self.visited)
    
    def evaluate_1hop(self, position: Tuple[int, int], 
                      goal: Tuple[int, int]) -> Dict:
        """Standard 1-hop evaluation with Manhattan distance"""
        neighbors = self._get_neighbors(position)
        
        best_action = None
        best_distance = float('inf')
        
        for neighbor in neighbors:
            if neighbor not in self.visited:
                dist = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                if dist < best_distance:
                    best_distance = dist
                    best_action = neighbor
        
        return {
            'action': best_action,
            'distance': best_distance,
            'type': '1hop'
        }
    
    def evaluate_2hop(self, position: Tuple[int, int], 
                      goal: Tuple[int, int]) -> Dict:
        """2-hop structural evaluation for better navigation"""
        
        # Check if entering dead-end using 2-hop lookahead
        dead_end_score = self._evaluate_deadend_2hop(position)
        
        if dead_end_score > 0.7:
            # Find backtrack point
            backtrack_point = self._find_backtrack_junction()
            if backtrack_point:
                return {
                    'action': 'backtrack',
                    'target': backtrack_point,
                    'reason': '2hop_deadend',
                    'score': dead_end_score,
                    'type': '2hop'
                }
        
        # Check for better path using 2-hop analysis
        better_path = self._find_better_path_2hop(position, goal)
        if better_path:
            return {
                'action': better_path,
                'reason': '2hop_better_path',
                'type': '2hop'
            }
        
        # Default to 1-hop
        return self.evaluate_1hop(position, goal)
    
    def _evaluate_deadend_2hop(self, position: Tuple[int, int]) -> float:
        """Evaluate dead-end probability using 2-hop BFS"""
        
        if position in self.dead_end_cache:
            return 1.0
        
        # 2-hop BFS
        visited_bfs = {position}
        queue = deque([(position, 0)])
        exits_found = 0
        positions_at_2hop = []
        
        while queue:
            pos, depth = queue.popleft()
            
            if depth == 2:
                positions_at_2hop.append(pos)
                continue
            
            neighbors = self._get_neighbors(pos)
            unvisited = [n for n in neighbors if n not in self.visited]
            
            if depth == 1 and len(unvisited) > 1:
                exits_found += 1
            
            for neighbor in neighbors:
                if neighbor not in visited_bfs and neighbor not in self.visited:
                    visited_bfs.add(neighbor)
                    queue.append((neighbor, depth + 1))
        
        # Calculate dead-end score
        if len(positions_at_2hop) == 0:
            score = 1.0  # Complete dead-end
        else:
            # Score based on connectivity at 2-hop
            avg_exits = exits_found / max(1, len(positions_at_2hop))
            score = 1.0 - min(1.0, avg_exits / 2.0)
        
        if score > 0.8:
            # Cache dead-end positions
            for pos in visited_bfs:
                self.dead_end_cache.add(pos)
        
        return score
    
    def _find_better_path_2hop(self, position: Tuple[int, int], 
                               goal: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find better path using 2-hop evaluation"""
        
        neighbors = self._get_neighbors(position)
        unvisited = [n for n in neighbors if n not in self.visited]
        
        if len(unvisited) <= 1:
            return None
        
        best_neighbor = None
        best_score = float('-inf')
        
        for neighbor in unvisited:
            # Evaluate 2-hop potential
            score = self._evaluate_path_potential(neighbor, goal)
            if score > best_score:
                best_score = score
                best_neighbor = neighbor
        
        # Only return if significantly better than greedy
        greedy = self.evaluate_1hop(position, goal)['action']
        if best_neighbor and best_neighbor != greedy and best_score > 0.5:
            return best_neighbor
        
        return None
    
    def _evaluate_path_potential(self, position: Tuple[int, int], 
                                goal: Tuple[int, int]) -> float:
        """Evaluate path potential at 2-hop distance"""
        
        # Simple 2-hop connectivity score
        visited_temp = {position}
        queue = deque([(position, 0)])
        reachable_2hop = 0
        goal_direction_bonus = 0
        
        while queue:
            pos, depth = queue.popleft()
            
            if depth == 2:
                # Check if closer to goal
                dist = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
                base_dist = abs(position[0] - goal[0]) + abs(position[1] - goal[1])
                if dist < base_dist:
                    goal_direction_bonus += 1
                continue
            
            for neighbor in self._get_neighbors(pos):
                if neighbor not in visited_temp and neighbor not in self.visited:
                    visited_temp.add(neighbor)
                    reachable_2hop += 1
                    queue.append((neighbor, depth + 1))
        
        # Combined score
        connectivity_score = min(1.0, reachable_2hop / 10.0)
        direction_score = min(1.0, goal_direction_bonus / 3.0)
        
        return 0.7 * connectivity_score + 0.3 * direction_score
    
    def _find_backtrack_junction(self) -> Optional[Tuple[int, int]]:
        """Find good junction to backtrack to"""
        
        # Look for junction with unexplored paths
        for i in range(len(self.path) - 1, max(0, len(self.path) - 50), -1):
            pos = self.path[i]
            unvisited_count = self._count_unvisited_neighbors(pos)
            
            if unvisited_count >= 2:
                return pos
            elif unvisited_count == 1 and i < len(self.path) - 20:
                return pos
        
        # Fallback to earlier position
        if len(self.path) > 10:
            return self.path[len(self.path) // 2]
        
        return None
    
    def navigate(self, start: Tuple[int, int], goal: Tuple[int, int],
                use_2hop: bool = True, max_steps: int = 3000) -> Dict:
        """Navigate maze with optional 2-hop evaluation"""
        
        print(f"Starting navigation: {start} -> {goal}, 2-hop={use_2hop}")
        
        self.visited = {start}
        self.path = [start]
        self.decision_history = []
        position = start
        
        steps = 0
        backtrack_count = 0
        two_hop_decisions = 0
        
        start_time = time.time()
        
        while position != goal and steps < max_steps:
            # Evaluate next action
            if use_2hop:
                decision = self.evaluate_2hop(position, goal)
            else:
                decision = self.evaluate_1hop(position, goal)
            
            # Count 2-hop decisions
            if decision.get('type') == '2hop':
                two_hop_decisions += 1
            
            # Log progress every 100 steps
            if steps % 100 == 0:
                dist = abs(position[0] - goal[0]) + abs(position[1] - goal[1])
                print(f"Step {steps}: pos={position}, dist={dist}, "
                      f"visited={len(self.visited)}, 2hop_decisions={two_hop_decisions}")
            
            self.decision_history.append({
                'position': position,
                'decision': decision,
                'step': steps
            })
            
            # Execute action
            if decision.get('action') == 'backtrack':
                backtrack_count += 1
                target = decision.get('target')
                if target and target in self.path:
                    idx = self.path.index(target)
                    # Mark dead-end positions
                    for p in self.path[idx+1:]:
                        self.dead_end_cache.add(p)
                    self.path = self.path[:idx + 1]
                    position = target
            else:
                # Normal movement
                if decision['action'] and decision['action'] not in self.visited:
                    position = decision['action']
                    self.visited.add(position)
                    self.path.append(position)
                else:
                    # Need to backtrack
                    backtrack_count += 1
                    if len(self.path) > 1:
                        self.path.pop()
                        position = self.path[-1]
                    else:
                        break
            
            steps += 1
        
        elapsed_time = time.time() - start_time
        success = position == goal
        
        print(f"\nNavigation complete: success={success}, steps={steps}, "
              f"time={elapsed_time:.2f}s")
        print(f"Backtracks: {backtrack_count}, 2-hop decisions: {two_hop_decisions}")
        print(f"Dead-ends found: {len(self.dead_end_cache)}")
        
        return {
            'success': success,
            'steps': steps,
            'path_length': len(self.path),
            'backtrack_count': backtrack_count,
            'visited_count': len(self.visited),
            'dead_ends_found': len(self.dead_end_cache),
            'two_hop_decisions': two_hop_decisions,
            'elapsed_time': elapsed_time,
            'final_position': position,
            'goal': goal,
            'efficiency': len(self.visited) / steps * 100 if steps > 0 else 0
        }


def generate_50x50_maze(seed: int = 42) -> np.ndarray:
    """Generate the same 50x50 maze used in null_goal experiment"""
    np.random.seed(seed)
    
    size = 50
    maze = np.ones((size, size), dtype=int)
    
    # Recursive backtracker algorithm
    def carve_passages(cx, cy):
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        np.random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < size and 0 <= ny < size and maze[nx, ny] == 1:
                maze[cx + dx // 2, cy + dy // 2] = 0
                maze[nx, ny] = 0
                carve_passages(nx, ny)
    
    # Start from (1, 1)
    maze[1, 1] = 0
    carve_passages(1, 1)
    
    # Ensure start and goal are clear
    maze[1, 1] = 0
    maze[size-2, size-2] = 0
    
    # Add some additional paths for complexity
    for _ in range(size):
        x = np.random.randint(1, size-1)
        y = np.random.randint(1, size-1)
        if maze[x, y] == 1:
            # Check if creating path doesn't break walls too much
            neighbors = sum(1 for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]
                          if 0 <= x+dx < size and 0 <= y+dy < size and maze[x+dx, y+dy] == 0)
            if neighbors >= 2:
                maze[x, y] = 0
    
    return maze


def visualize_comparison(maze: np.ndarray, result_1hop: Dict, result_2hop: Dict):
    """Visualize comparison between 1-hop and 2-hop navigation"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Maze structure
    axes[0].imshow(maze, cmap='binary', interpolation='nearest')
    axes[0].set_title("50x50 Maze Structure")
    axes[0].plot(1, 1, 'go', markersize=8, label='Start')
    axes[0].plot(48, 48, 'ro', markersize=8, label='Goal')
    axes[0].legend()
    axes[0].axis('off')
    
    # 1-hop result
    axes[1].imshow(maze, cmap='binary', interpolation='nearest')
    
    # Create visited heatmap
    visited_map = np.zeros_like(maze, dtype=float)
    for i in range(50):
        for j in range(50):
            if (i, j) in result_1hop.get('visited', set()):
                visited_map[i, j] = 0.5
    
    axes[1].imshow(visited_map, cmap='Blues', alpha=0.5, interpolation='nearest')
    axes[1].set_title(f"1-hop: Steps={result_1hop['steps']}, "
                     f"Efficiency={result_1hop['efficiency']:.1f}%")
    axes[1].plot(1, 1, 'go', markersize=8)
    
    final_pos = result_1hop.get('final_position', (1, 1))
    if result_1hop['success']:
        axes[1].plot(48, 48, 'go', markersize=8)
    else:
        axes[1].plot(final_pos[1], final_pos[0], 'rx', markersize=10)
        axes[1].plot(48, 48, 'ro', markersize=8)
    
    axes[1].axis('off')
    
    # 2-hop result
    axes[2].imshow(maze, cmap='binary', interpolation='nearest')
    
    # Visited heatmap for 2-hop
    visited_map_2hop = np.zeros_like(maze, dtype=float)
    dead_end_map = np.zeros_like(maze, dtype=float)
    
    # Get visited positions from path since we might not have full visited set
    if 'path' in result_2hop:
        for pos in result_2hop['path']:
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                visited_map_2hop[pos[0], pos[1]] = 0.5
    
    # Mark dead-ends
    for pos in result_2hop.get('dead_end_cache', set()):
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            dead_end_map[pos[0], pos[1]] = 1.0
    
    axes[2].imshow(visited_map_2hop, cmap='Blues', alpha=0.5, interpolation='nearest')
    if np.any(dead_end_map > 0):
        axes[2].imshow(dead_end_map, cmap='Reds', alpha=0.3, interpolation='nearest')
    
    axes[2].set_title(f"2-hop: Steps={result_2hop['steps']}, "
                     f"Efficiency={result_2hop['efficiency']:.1f}%, "
                     f"2hop-decisions={result_2hop['two_hop_decisions']}")
    axes[2].plot(1, 1, 'go', markersize=8)
    
    final_pos_2hop = result_2hop.get('final_position', (1, 1))
    if result_2hop['success']:
        axes[2].plot(48, 48, 'go', markersize=8)
    else:
        axes[2].plot(final_pos_2hop[1], final_pos_2hop[0], 'rx', markersize=10)
        axes[2].plot(48, 48, 'ro', markersize=8)
    
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'maze_50x50_2hop_comparison_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {filename}")
    
    # Also save results
    results = {
        'timestamp': timestamp,
        'maze_size': 50,
        '1hop': result_1hop,
        '2hop': result_2hop,
        'improvement': {
            'steps_reduction': (1 - result_2hop['steps']/result_1hop['steps'])*100 if result_1hop['steps'] > 0 else 0,
            'efficiency_gain': result_2hop['efficiency'] - result_1hop['efficiency'],
            'backtrack_reduction': (1 - result_2hop['backtrack_count']/max(1, result_1hop['backtrack_count']))*100
        }
    }
    
    with open(f'maze_50x50_2hop_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)


def main():
    """Run 50x50 maze 2-hop evaluation experiment"""
    
    print("="*70)
    print("50x50 MAZE 2-HOP EVALUATION EXPERIMENT")
    print("="*70)
    
    # Generate maze
    print("\nGenerating 50x50 maze...")
    maze = generate_50x50_maze(seed=42)
    
    start = (1, 1)
    goal = (48, 48)
    
    # Test 1-hop
    print("\n--- Testing 1-hop navigation ---")
    navigator_1hop = FixedMaze2HopNavigator(maze)
    result_1hop = navigator_1hop.navigate(start, goal, use_2hop=False, max_steps=3000)
    
    # Test 2-hop
    print("\n--- Testing 2-hop navigation ---")
    navigator_2hop = FixedMaze2HopNavigator(maze)
    result_2hop = navigator_2hop.navigate(start, goal, use_2hop=True, max_steps=3000)
    
    # Results summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n1-hop Navigation:")
    print(f"  Success: {result_1hop['success']}")
    print(f"  Steps: {result_1hop['steps']}")
    print(f"  Visited cells: {result_1hop['visited_count']}")
    print(f"  Efficiency: {result_1hop['efficiency']:.1f}%")
    print(f"  Backtracks: {result_1hop['backtrack_count']}")
    
    print(f"\n2-hop Navigation:")
    print(f"  Success: {result_2hop['success']}")
    print(f"  Steps: {result_2hop['steps']}")
    print(f"  Visited cells: {result_2hop['visited_count']}")
    print(f"  Efficiency: {result_2hop['efficiency']:.1f}%")
    print(f"  Backtracks: {result_2hop['backtrack_count']}")
    print(f"  2-hop decisions: {result_2hop['two_hop_decisions']}")
    print(f"  Dead-ends detected: {result_2hop['dead_ends_found']}")
    
    if result_1hop['steps'] > 0 and result_2hop['steps'] > 0:
        print(f"\nImprovement:")
        print(f"  Steps reduction: {(1 - result_2hop['steps']/result_1hop['steps'])*100:.1f}%")
        print(f"  Efficiency gain: {result_2hop['efficiency'] - result_1hop['efficiency']:.1f}%")
        if result_1hop['backtrack_count'] > 0:
            print(f"  Backtrack reduction: {(1 - result_2hop['backtrack_count']/result_1hop['backtrack_count'])*100:.1f}%")
    
    # Visualize
    visualize_comparison(maze, result_1hop, result_2hop)
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()