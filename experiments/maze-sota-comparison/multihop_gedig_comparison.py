#!/usr/bin/env python3
"""
Multi-hop geDIG Navigator Comparison
====================================

Tests 1-hop, 2-hop, and 3-hop evaluation to see if higher-order
structural evaluation improves coverage and efficiency.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
from datetime import datetime
import json

@dataclass
class Episode:
    """Episode with normalized vector representation"""
    position: Tuple[int, int]
    action: Optional[str]
    result: str
    visit_count: int
    vector: np.ndarray
    timestamp: int

class MultiHopGeDIGNavigator:
    """Navigator with configurable n-hop evaluation"""
    
    def __init__(self, maze: np.ndarray, hop_count: int = 1):
        self.maze = maze
        self.height, self.width = maze.shape
        self.position = (1, 1)
        self.visited = {(1, 1)}
        self.path = [(1, 1)]
        self.hop_count = hop_count
        
        # Episode memory
        self.episodes: List[Episode] = []
        self.position_visits = {(1, 1): 1}
        self.episode_counter = 0
        
        # Visual memory
        self.visual_memory = {}
        
        # Statistics
        self.coverage_over_time = []
        self.exploration_choices = 0
        self.exploitation_choices = 0
        
        # Initialize
        self._update_visual_memory(1, 1)
        
    def _update_visual_memory(self, x: int, y: int):
        """Update visual memory from current position"""
        if (x, y) not in self.visual_memory:
            self.visual_memory[(x, y)] = {}
        
        for action, (dx, dy) in [('up', (0, -1)), ('right', (1, 0)), 
                                 ('down', (0, 1)), ('left', (-1, 0))]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                self.visual_memory[(x, y)][action] = self.maze[ny, nx] == 0
    
    def _normalize_position(self, pos: Tuple[int, int]) -> np.ndarray:
        """Normalize position to [-1, 1]"""
        x, y = pos
        norm_x = (x / self.width) * 2 - 1
        norm_y = (y / self.height) * 2 - 1
        return np.array([norm_x, norm_y])
    
    def _calculate_n_hop_value(self, next_pos: Tuple[int, int], n: int) -> float:
        """Calculate n-hop evaluation value"""
        if n <= 0 or next_pos in self.visited:
            return -1.0
        
        value = 0.0
        visit_count = self.position_visits.get(next_pos, 0)
        
        # Base value: exploration bonus
        if visit_count == 0:
            value += 2.0
            self.exploration_choices += 1
        else:
            value += 1.0 / (1 + visit_count)
            self.exploitation_choices += 1
        
        # n-hop connectivity bonus
        reachable = self._get_reachable_positions(next_pos, n)
        
        # Count unexplored positions reachable in n hops
        unexplored_reachable = sum(1 for pos in reachable if pos not in self.visited)
        value += unexplored_reachable * 0.5
        
        # Bonus for connecting distant explored regions
        if n >= 2:
            connection_bonus = self._calculate_connection_bonus(next_pos, reachable)
            value += connection_bonus
        
        return value
    
    def _get_reachable_positions(self, start_pos: Tuple[int, int], 
                                max_hops: int) -> set:
        """Get all positions reachable within max_hops"""
        reachable = {start_pos}
        current_layer = {start_pos}
        
        for hop in range(max_hops):
            next_layer = set()
            for pos in current_layer:
                x, y = pos
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.width and 
                        0 <= ny < self.height and 
                        self.maze[ny, nx] == 0 and
                        (nx, ny) not in reachable):
                        next_layer.add((nx, ny))
            reachable.update(next_layer)
            current_layer = next_layer
            
        return reachable
    
    def _calculate_connection_bonus(self, pos: Tuple[int, int], 
                                  reachable: set) -> float:
        """Bonus for connecting separate explored regions"""
        # Find visited positions in reachable set
        visited_reachable = [p for p in reachable if p in self.visited and p != pos]
        
        if len(visited_reachable) < 2:
            return 0.0
        
        # Check if this position connects previously disconnected regions
        # Simplified: count distinct regions by checking path connectivity
        regions = []
        for vp in visited_reachable:
            connected = False
            for region in regions:
                if any(self._are_adjacent(vp, rp) for rp in region):
                    region.add(vp)
                    connected = True
                    break
            if not connected:
                regions.append({vp})
        
        # Bonus for connecting multiple regions
        if len(regions) > 1:
            return len(regions) * 0.3
        
        return 0.0
    
    def _are_adjacent(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        """Check if two positions are adjacent"""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1
    
    def decide_action(self) -> str:
        """Decide action using n-hop evaluation"""
        current_pos = self.position
        visual_info = self.visual_memory.get(current_pos, {})
        
        action_values = {}
        
        for action, (dx, dy) in [('up', (0, -1)), ('right', (1, 0)), 
                                ('down', (0, 1)), ('left', (-1, 0))]:
            # Skip walls
            if action in visual_info and not visual_info[action]:
                continue
            
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            # Skip invalid positions
            if not (0 <= next_pos[0] < self.width and 
                   0 <= next_pos[1] < self.height):
                continue
            
            # Calculate n-hop value
            value = self._calculate_n_hop_value(next_pos, self.hop_count)
            action_values[action] = value
        
        if not action_values:
            return 'backtrack'
        
        # Softmax selection with temperature
        actions = list(action_values.keys())
        values = np.array(list(action_values.values()))
        
        # Add small noise for exploration
        values += np.random.normal(0, 0.1, len(values))
        
        temperature = 0.3
        exp_values = np.exp(values / temperature)
        probs = exp_values / exp_values.sum()
        
        return np.random.choice(actions, p=probs)
    
    def navigate(self, goal: Tuple[int, int], max_steps: int = 5000) -> Dict:
        """Navigate maze using n-hop evaluation"""
        
        print(f"\n{self.hop_count}-hop geDIG Navigation")
        print(f"Start: {self.position}, Goal: {goal}")
        
        steps = 0
        backtrack_count = 0
        
        # Track walkable cells for coverage calculation
        walkable_cells = sum(1 for y in range(self.height) 
                           for x in range(self.width) 
                           if self.maze[y, x] == 0)
        
        start_time = time.time()
        
        while self.position != goal and steps < max_steps:
            # Track coverage
            if steps % 50 == 0:
                coverage = len(self.visited) / walkable_cells * 100
                self.coverage_over_time.append((steps, coverage))
            
            # Progress report
            if steps % 500 == 0:
                dist = abs(self.position[0] - goal[0]) + abs(self.position[1] - goal[1])
                coverage = len(self.visited) / walkable_cells * 100
                print(f"Step {steps}: pos={self.position}, dist={dist}, "
                      f"coverage={coverage:.1f}%")
            
            # Decide action
            action = self.decide_action()
            
            if action == 'backtrack':
                backtrack_count += 1
                if len(self.path) > 1:
                    self.path.pop()
                    self.position = self.path[-1]
            else:
                # Execute action
                dx, dy = {'up': (0, -1), 'right': (1, 0), 
                         'down': (0, 1), 'left': (-1, 0)}[action]
                next_pos = (self.position[0] + dx, self.position[1] + dy)
                
                # Record episode
                visit_count = self.position_visits.get(next_pos, 0)
                vector = np.concatenate([
                    self._normalize_position(self.position),
                    self._normalize_position(next_pos),
                    [visit_count / 10.0]  # Normalized visit count
                ])
                
                episode = Episode(
                    position=self.position,
                    action=action,
                    result='success',
                    visit_count=visit_count,
                    vector=vector,
                    timestamp=steps
                )
                self.episodes.append(episode)
                
                # Move
                self.position = next_pos
                self.visited.add(next_pos)
                self.path.append(next_pos)
                self.position_visits[next_pos] = visit_count + 1
                
                # Update visual memory
                self._update_visual_memory(next_pos[0], next_pos[1])
            
            steps += 1
        
        elapsed_time = time.time() - start_time
        success = self.position == goal
        final_coverage = len(self.visited) / walkable_cells * 100
        
        print(f"\nNavigation complete: success={success}, steps={steps}")
        print(f"Coverage: {final_coverage:.1f}%, Efficiency: {len(self.visited)/steps*100:.1f}%")
        print(f"Exploration: {self.exploration_choices}, Exploitation: {self.exploitation_choices}")
        
        return {
            'success': success,
            'steps': steps,
            'visited_count': len(self.visited),
            'coverage': final_coverage,
            'efficiency': len(self.visited) / steps * 100 if steps > 0 else 0,
            'backtrack_count': backtrack_count,
            'elapsed_time': elapsed_time,
            'coverage_over_time': self.coverage_over_time,
            'exploration_ratio': self.exploration_choices / (self.exploration_choices + self.exploitation_choices)
        }


def compare_multi_hop():
    """Compare 1-hop, 2-hop, and 3-hop navigation"""
    
    # Generate maze
    np.random.seed(42)
    size = 50
    maze = np.ones((size, size), dtype=int)
    
    # Create maze with recursive backtracker
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
    
    # Add some loops for interesting topology
    for _ in range(size):
        x = np.random.randint(2, size-2)
        y = np.random.randint(2, size-2)
        if maze[y, x] == 1:
            neighbors = sum(1 for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]
                          if 0 <= x+dx < size and 0 <= y+dy < size and maze[y+dy, x+dx] == 0)
            if neighbors >= 2:
                maze[y, x] = 0
    
    print("="*70)
    print("MULTI-HOP geDIG NAVIGATION COMPARISON")
    print("="*70)
    
    goal = (size-2, size-2)
    results = {}
    
    # Test different hop counts
    for hop_count in [1, 2, 3]:
        print(f"\n{'='*30} {hop_count}-HOP {'='*30}")
        navigator = MultiHopGeDIGNavigator(maze, hop_count=hop_count)
        result = navigator.navigate(goal, max_steps=5000)
        results[f"{hop_count}-hop"] = result
    
    # Display results
    print("\n" + "="*70)
    print("FINAL RESULTS COMPARISON")
    print("="*70)
    
    print("\n| Method | Success | Steps | Coverage | Efficiency | Exploration |")
    print("|--------|---------|-------|----------|------------|-------------|")
    
    for method, result in results.items():
        print(f"| {method:6} | {str(result['success']):7} | {result['steps']:5} | "
              f"{result['coverage']:7.1f}% | {result['efficiency']:9.1f}% | "
              f"{result['exploration_ratio']*100:10.1f}% |")
    
    # Calculate improvements
    if results['1-hop']['steps'] > 0:
        print("\nImprovements over 1-hop:")
        for hop_count in [2, 3]:
            key = f"{hop_count}-hop"
            step_improvement = (1 - results[key]['steps']/results['1-hop']['steps']) * 100
            coverage_improvement = results[key]['coverage'] - results['1-hop']['coverage']
            efficiency_improvement = results[key]['efficiency'] - results['1-hop']['efficiency']
            
            print(f"\n{hop_count}-hop improvements:")
            print(f"  Steps: {step_improvement:+.1f}%")
            print(f"  Coverage: {coverage_improvement:+.1f}%")
            print(f"  Efficiency: {efficiency_improvement:+.1f}%")
    
    # Plot coverage over time
    plt.figure(figsize=(10, 6))
    for method, result in results.items():
        if result['coverage_over_time']:
            steps, coverage = zip(*result['coverage_over_time'])
            plt.plot(steps, coverage, label=method, linewidth=2)
    
    plt.xlabel('Steps')
    plt.ylabel('Coverage (%)')
    plt.title('Coverage Rate Over Time by Hop Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('multihop_coverage_comparison.png', dpi=150)
    plt.close()
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'multihop_results_{timestamp}.json', 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'maze_size': size,
            'results': {k: {kk: vv for kk, vv in v.items() if kk != 'coverage_over_time'} 
                       for k, v in results.items()}
        }, f, indent=2)
    
    print(f"\nResults saved to multihop_results_{timestamp}.json")
    print("Coverage plot saved to multihop_coverage_comparison.png")
    
    return results


if __name__ == "__main__":
    results = compare_multi_hop()