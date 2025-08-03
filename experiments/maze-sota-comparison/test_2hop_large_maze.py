#!/usr/bin/env python3
"""
Test 2-hop evaluation in large complex maze (50x50)
==================================================

Compare 1-hop vs 2-hop navigation in challenging mazes
with multiple dead-ends and complex branching.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
import time
from datetime import datetime


class LargeMaze2HopNavigator:
    """Advanced maze navigator with 2-hop structural evaluation for large mazes"""
    
    def __init__(self, maze: np.ndarray, sparse_graph: bool = True):
        self.maze = maze
        self.height, self.width = maze.shape
        self.visited = set()
        self.path = []
        self.decision_history = []
        self.dead_end_cache = set()  # Cache detected dead-ends
        
        # For large mazes, use sparse representation
        self.sparse_graph = sparse_graph
        if not sparse_graph:
            self.maze_graph = self._build_maze_graph()
        else:
            self.junction_graph = self._build_junction_graph()
        
    def _build_maze_graph(self) -> nx.Graph:
        """Build full graph representation (memory intensive for 50x50)"""
        print("Building full maze graph...")
        G = nx.Graph()
        
        # Add nodes for all walkable cells
        walkable = []
        for i in range(self.height):
            for j in range(self.width):
                if self.maze[i, j] == 0:  # Walkable
                    G.add_node((i, j))
                    walkable.append((i, j))
        
        print(f"Found {len(walkable)} walkable cells")
        
        # Add edges between adjacent walkable cells
        for i, j in walkable:
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                if (ni, nj) in G:
                    G.add_edge((i, j), (ni, nj))
        
        print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def _build_junction_graph(self) -> nx.Graph:
        """Build sparse graph of junctions only (memory efficient)"""
        print("Building junction graph...")
        junctions = self._find_junctions()
        G = nx.Graph()
        
        # Add junction nodes
        for j in junctions:
            G.add_node(j)
        
        # Connect junctions that are reachable
        for i, j1 in enumerate(junctions):
            for j2 in junctions[i+1:]:
                if self._junctions_connected(j1, j2):
                    dist = abs(j1[0] - j2[0]) + abs(j1[1] - j2[1])
                    G.add_edge(j1, j2, weight=dist)
        
        print(f"Junction graph has {G.number_of_nodes()} nodes")
        return G
    
    def _find_junctions(self) -> List[Tuple[int, int]]:
        """Find junction points (3+ exits) and dead-ends (1 exit)"""
        junctions = []
        
        for i in range(self.height):
            for j in range(self.width):
                if self.maze[i, j] == 0:
                    exits = sum(1 for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]
                               if 0 <= i+di < self.height and 0 <= j+dj < self.width 
                               and self.maze[i+di, j+dj] == 0)
                    if exits >= 3 or exits == 1:  # Junction or dead-end
                        junctions.append((i, j))
        
        return junctions
    
    def _junctions_connected(self, j1: Tuple[int, int], 
                           j2: Tuple[int, int]) -> bool:
        """Check if two junctions are connected by a path"""
        # Simple BFS with early termination
        if abs(j1[0] - j2[0]) + abs(j1[1] - j2[1]) > 100:
            return False  # Too far
        
        visited = {j1}
        queue = deque([j1])
        
        while queue:
            pos = queue.popleft()
            if pos == j2:
                return True
            
            for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                ni, nj = pos[0] + di, pos[1] + dj
                next_pos = (ni, nj)
                
                if (next_pos not in visited and 
                    0 <= ni < self.height and 
                    0 <= nj < self.width and
                    self.maze[ni, nj] == 0):
                    visited.add(next_pos)
                    queue.append(next_pos)
        
        return False
    
    def evaluate_1hop(self, position: Tuple[int, int], 
                      goal: Tuple[int, int]) -> Dict:
        """Standard 1-hop evaluation with A* heuristic"""
        neighbors = self._get_neighbors(position)
        
        best_action = None
        best_score = float('inf')
        
        for neighbor in neighbors:
            if neighbor not in self.visited:
                # A* heuristic: f = g + h
                g = 1  # Cost to neighbor
                h = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])  # Manhattan distance
                score = g + h
                
                if score < best_score:
                    best_score = score
                    best_action = neighbor
        
        return {
            'action': best_action,
            'score': best_score,
            'type': '1hop'
        }
    
    def evaluate_2hop_advanced(self, position: Tuple[int, int], 
                              goal: Tuple[int, int]) -> Dict:
        """Advanced 2-hop evaluation with multiple strategies"""
        
        # Strategy 1: Dead-end detection via limited BFS
        if self._is_entering_deadend(position, goal):
            backtrack_point = self._find_smart_backtrack_point(position)
            return {
                'action': 'backtrack',
                'target': backtrack_point,
                'reason': 'dead_end_detected',
                'type': '2hop'
            }
        
        # Strategy 2: Loop detection
        if self._is_in_loop(position):
            escape_point = self._find_loop_escape(position)
            return {
                'action': 'backtrack',
                'target': escape_point,
                'reason': 'loop_detected',
                'type': '2hop'
            }
        
        # Strategy 3: Junction analysis
        junction_score = self._evaluate_junction_quality(position, goal)
        if junction_score < 0.3:  # Poor junction
            alt_junction = self._find_alternative_junction(position, goal)
            if alt_junction:
                return {
                    'action': 'seek_junction',
                    'target': alt_junction,
                    'reason': 'better_junction_available',
                    'type': '2hop'
                }
        
        # Default to 1-hop
        return self.evaluate_1hop(position, goal)
    
    def _is_entering_deadend(self, position: Tuple[int, int], 
                            goal: Tuple[int, int], 
                            max_depth: int = 15) -> bool:
        """Detect if path leads to dead-end using limited BFS"""
        
        # Check cache first
        if position in self.dead_end_cache:
            return True
        
        # Limited BFS to detect dead-ends
        visited = {position}
        queue = deque([(position, 0)])
        exits_found = 0
        goal_reachable = False
        
        while queue and len(visited) < max_depth:
            pos, depth = queue.popleft()
            
            if pos == goal:
                goal_reachable = True
                break
            
            neighbors = self._get_neighbors(pos)
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            
            if len(unvisited_neighbors) == 0 and depth > 0:
                # Dead end found
                self.dead_end_cache.add(pos)
            elif len(unvisited_neighbors) > 1:
                exits_found += 1
            
            for neighbor in unvisited_neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        
        # It's a dead-end if no exits found and goal not reachable
        is_deadend = exits_found == 0 and not goal_reachable
        
        if is_deadend:
            # Cache this path as dead-end
            for v in visited:
                self.dead_end_cache.add(v)
        
        return is_deadend
    
    def _is_in_loop(self, position: Tuple[int, int], 
                    window: int = 20) -> bool:
        """Detect if navigator is stuck in a loop"""
        if len(self.path) < window:
            return False
        
        recent_positions = self.path[-window:]
        position_counts = {}
        
        for pos in recent_positions:
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        # If any position visited more than twice recently, it's a loop
        return any(count > 2 for count in position_counts.values())
    
    def _evaluate_junction_quality(self, position: Tuple[int, int], 
                                  goal: Tuple[int, int]) -> float:
        """Evaluate quality of current junction (0-1, higher is better)"""
        neighbors = self._get_neighbors(position)
        unvisited = [n for n in neighbors if n not in self.visited]
        
        if len(unvisited) == 0:
            return 0.0
        
        # Quality based on:
        # 1. Number of unexplored paths
        # 2. Direction towards goal
        # 3. Not being in dead-end cache
        
        quality = len(unvisited) / 4.0  # Normalize by max possible
        
        # Bonus for paths toward goal
        goal_direction = (goal[0] - position[0], goal[1] - position[1])
        goal_dir_norm = np.sqrt(goal_direction[0]**2 + goal_direction[1]**2)
        
        if goal_dir_norm > 0:
            for neighbor in unvisited:
                neighbor_dir = (neighbor[0] - position[0], neighbor[1] - position[1])
                dot_product = (goal_direction[0] * neighbor_dir[0] + 
                             goal_direction[1] * neighbor_dir[1]) / goal_dir_norm
                if dot_product > 0:
                    quality += 0.25
        
        # Penalty for dead-end paths
        for neighbor in unvisited:
            if neighbor in self.dead_end_cache:
                quality -= 0.5
        
        return max(0, min(1, quality))
    
    def _find_smart_backtrack_point(self, position: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find intelligent backtrack point (junction with unexplored paths)"""
        
        # Look for recent junctions with unexplored paths
        for i in range(len(self.path) - 1, -1, -1):
            pos = self.path[i]
            unvisited = self._count_unvisited_neighbors(pos)
            
            # Prioritize junctions with multiple unexplored paths
            if unvisited >= 2:
                return pos
            elif unvisited == 1 and i < len(self.path) - 10:
                # Accept single unexplored if we've gone far enough
                return pos
        
        # Last resort: go back quarter of the path
        if len(self.path) > 4:
            return self.path[len(self.path) // 4]
        
        return None
    
    def _find_alternative_junction(self, position: Tuple[int, int], 
                                  goal: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find a better junction nearby"""
        
        # BFS to find nearby junctions
        visited = {position}
        queue = deque([(position, 0)])
        junctions_found = []
        
        while queue and len(junctions_found) < 5:
            pos, dist = queue.popleft()
            
            if dist > 10:  # Don't search too far
                continue
            
            # Check if it's a junction
            exits = self._count_unvisited_neighbors(pos)
            if exits >= 2 and pos != position:
                quality = self._evaluate_junction_quality(pos, goal)
                junctions_found.append((pos, quality))
            
            # Continue BFS
            for neighbor in self._get_neighbors(pos):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        # Return best junction found
        if junctions_found:
            junctions_found.sort(key=lambda x: x[1], reverse=True)
            return junctions_found[0][0]
        
        return None
    
    def _find_loop_escape(self, position: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find escape point from detected loop"""
        
        # Find earliest junction in recent path that has unexplored paths
        recent_start = max(0, len(self.path) - 30)
        
        for i in range(len(self.path) - 1, recent_start - 1, -1):
            pos = self.path[i]
            if self._count_unvisited_neighbors(pos) > 0:
                return pos
        
        # Otherwise, jump back significantly
        if len(self.path) > 20:
            return self.path[len(self.path) // 2]
        
        return None
    
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
        """Count unvisited neighbors of a position"""
        return sum(1 for n in self._get_neighbors(position) if n not in self.visited)
    
    def navigate(self, start: Tuple[int, int], goal: Tuple[int, int],
                use_2hop: bool = True, max_steps: int = 5000) -> Dict:
        """Navigate maze with optional 2-hop evaluation"""
        
        print(f"Starting navigation: {start} -> {goal}, 2-hop={use_2hop}")
        
        self.visited = {start}
        self.path = [start]
        self.decision_history = []
        position = start
        
        steps = 0
        backtrack_count = 0
        last_progress_step = 0
        best_distance = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
        
        start_time = time.time()
        
        while position != goal and steps < max_steps:
            # Check progress
            current_distance = abs(goal[0] - position[0]) + abs(goal[1] - position[1])
            if current_distance < best_distance:
                best_distance = current_distance
                last_progress_step = steps
            
            # Evaluate next action
            if use_2hop:
                decision = self.evaluate_2hop_advanced(position, goal)
            else:
                decision = self.evaluate_1hop(position, goal)
            
            # Log important decisions
            if steps % 100 == 0 or decision.get('action') in ['backtrack', 'seek_junction']:
                print(f"Step {steps}: pos={position}, dist_to_goal={current_distance}, "
                      f"decision={decision.get('action', 'move')}")
            
            self.decision_history.append({
                'position': position,
                'decision': decision,
                'step': steps,
                'distance_to_goal': current_distance
            })
            
            # Execute action
            if decision.get('action') == 'backtrack':
                backtrack_count += 1
                target = decision.get('target')
                if target and target in self.path:
                    # Smart backtrack
                    idx = self.path.index(target)
                    # Mark some positions as visited to avoid revisiting
                    for p in self.path[idx+1:]:
                        if self._count_unvisited_neighbors(p) == 0:
                            self.dead_end_cache.add(p)
                    self.path = self.path[:idx + 1]
                    position = target
                else:
                    # Fallback
                    if len(self.path) > 1:
                        self.path.pop()
                        position = self.path[-1]
                        
            elif decision.get('action') == 'seek_junction':
                # Try to reach alternative junction
                target = decision.get('target')
                if target:
                    # Simple pathfinding to target
                    next_step = self._step_toward(position, target)
                    if next_step and next_step not in self.visited:
                        position = next_step
                        self.visited.add(position)
                        self.path.append(position)
                    else:
                        # Can't reach, fallback to 1-hop
                        decision = self.evaluate_1hop(position, goal)
                        if decision['action']:
                            position = decision['action']
                            self.visited.add(position)
                            self.path.append(position)
                            
            else:
                # Normal movement
                if decision['action'] and decision['action'] not in self.visited:
                    position = decision['action']
                    self.visited.add(position)
                    self.path.append(position)
                else:
                    # All neighbors visited, need to backtrack
                    backtrack_count += 1
                    if len(self.path) > 1:
                        self.path.pop()
                        position = self.path[-1]
                    else:
                        break  # Stuck at start
            
            steps += 1
            
            # Check if stuck (no progress for many steps)
            if steps - last_progress_step > 500:
                print(f"No progress for 500 steps, terminating...")
                break
        
        elapsed_time = time.time() - start_time
        success = position == goal
        
        print(f"Navigation complete: success={success}, steps={steps}, time={elapsed_time:.2f}s")
        
        return {
            'success': success,
            'steps': steps,
            'path_length': len(self.path),
            'backtrack_count': backtrack_count,
            'visited_count': len(self.visited),
            'dead_ends_found': len(self.dead_end_cache),
            'best_distance': best_distance,
            'final_distance': current_distance,
            'elapsed_time': elapsed_time,
            'path': self.path.copy() if len(self.path) < 1000 else self.path[-100:],  # Limit size
            'decision_history': self.decision_history[-100:]  # Keep last 100 decisions
        }
    
    def _step_toward(self, current: Tuple[int, int], 
                     target: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Take one step toward target"""
        neighbors = self._get_neighbors(current)
        
        best_neighbor = None
        best_distance = float('inf')
        
        for neighbor in neighbors:
            dist = abs(neighbor[0] - target[0]) + abs(neighbor[1] - target[1])
            if dist < best_distance:
                best_distance = dist
                best_neighbor = neighbor
        
        return best_neighbor


def generate_complex_maze(size: int = 50, complexity: float = 0.4, 
                         density: float = 0.4) -> np.ndarray:
    """Generate a complex maze using recursive division"""
    
    # Initialize
    maze = np.ones((size, size), dtype=int)
    
    # Create border
    maze[0, :] = maze[-1, :] = 1
    maze[:, 0] = maze[:, -1] = 1
    
    # Recursive division
    def divide(x1, y1, x2, y2):
        if x2 - x1 < 2 or y2 - y1 < 2:
            return
        
        # Create room
        for i in range(x1, x2 + 1):
            for j in range(y1, y2 + 1):
                maze[i, j] = 0
        
        # Add walls
        if x2 - x1 > y2 - y1:
            # Vertical wall
            if x2 - x1 > 3:  # Ensure we have room for wall
                wall_x = x1 + 2 + np.random.randint(x2 - x1 - 3)
                for j in range(y1, y2 + 1):
                    maze[wall_x, j] = 1
                # Add passages
                for _ in range(max(1, (y2 - y1) // 4)):
                    passage_y = y1 + np.random.randint(y2 - y1 + 1)
                    maze[wall_x, passage_y] = 0
                # Recurse
                divide(x1, y1, wall_x - 1, y2)
                divide(wall_x + 1, y1, x2, y2)
        else:
            # Horizontal wall
            if y2 - y1 > 3:  # Ensure we have room for wall
                wall_y = y1 + 2 + np.random.randint(y2 - y1 - 3)
                for i in range(x1, x2 + 1):
                    maze[i, wall_y] = 1
                # Add passages
                for _ in range(max(1, (x2 - x1) // 4)):
                    passage_x = x1 + np.random.randint(x2 - x1 + 1)
                    maze[passage_x, wall_y] = 0
                # Recurse
                divide(x1, y1, x2, wall_y - 1)
                divide(x1, wall_y + 1, x2, y2)
    
    # Start division
    divide(1, 1, size - 2, size - 2)
    
    # Add some random obstacles
    for _ in range(int(size * size * density * 0.1)):
        x = np.random.randint(1, size - 1)
        y = np.random.randint(1, size - 1)
        if (x, y) != (1, 1) and (x, y) != (size - 2, size - 2):
            maze[x, y] = 1
    
    # Ensure start and goal are clear
    maze[1, 1] = 0
    maze[size - 2, size - 2] = 0
    
    # Ensure connectivity (simple path along edges)
    for i in range(1, size // 2):
        maze[i, 1] = 0
        maze[size - 2, i] = 0
    
    return maze


def run_large_maze_experiment():
    """Run experiment on large complex maze"""
    
    print("="*70)
    print("LARGE MAZE EXPERIMENT (50x50)")
    print("="*70)
    
    # Generate complex maze
    print("Generating complex maze...")
    maze = generate_complex_maze(50, complexity=0.4, density=0.3)
    
    start = (1, 1)
    goal = (48, 48)
    
    # Test 1-hop
    print("\n--- Testing 1-hop navigation ---")
    navigator_1hop = LargeMaze2HopNavigator(maze, sparse_graph=True)
    result_1hop = navigator_1hop.navigate(start, goal, use_2hop=False, max_steps=5000)
    
    # Test 2-hop
    print("\n--- Testing 2-hop navigation ---")
    navigator_2hop = LargeMaze2HopNavigator(maze, sparse_graph=True)
    result_2hop = navigator_2hop.navigate(start, goal, use_2hop=True, max_steps=5000)
    
    # Compare results
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    
    print(f"\n1-hop Navigation:")
    print(f"  Success: {result_1hop['success']}")
    print(f"  Steps: {result_1hop['steps']}")
    print(f"  Path length: {result_1hop['path_length']}")
    print(f"  Backtracks: {result_1hop['backtrack_count']}")
    print(f"  Visited cells: {result_1hop['visited_count']}")
    print(f"  Final distance to goal: {result_1hop['final_distance']}")
    print(f"  Time: {result_1hop['elapsed_time']:.2f}s")
    
    print(f"\n2-hop Navigation:")
    print(f"  Success: {result_2hop['success']}")
    print(f"  Steps: {result_2hop['steps']}")
    print(f"  Path length: {result_2hop['path_length']}")
    print(f"  Backtracks: {result_2hop['backtrack_count']}")
    print(f"  Visited cells: {result_2hop['visited_count']}")
    print(f"  Dead-ends detected: {result_2hop['dead_ends_found']}")
    print(f"  Final distance to goal: {result_2hop['final_distance']}")
    print(f"  Time: {result_2hop['elapsed_time']:.2f}s")
    
    print(f"\nImprovement with 2-hop:")
    if result_1hop['steps'] > 0:
        print(f"  Steps reduction: {(1 - result_2hop['steps']/result_1hop['steps'])*100:.1f}%")
    if result_1hop['backtrack_count'] > 0:
        print(f"  Backtrack reduction: {(1 - result_2hop['backtrack_count']/result_1hop['backtrack_count'])*100:.1f}%")
    
    # Visualize results
    visualize_large_maze_results(maze, result_1hop, result_2hop)
    
    # Analyze 2-hop decisions
    print("\n2-hop Decision Analysis:")
    decisions_by_type = {}
    for decision in result_2hop['decision_history']:
        if decision['decision'].get('type') == '2hop':
            reason = decision['decision'].get('reason', 'unknown')
            decisions_by_type[reason] = decisions_by_type.get(reason, 0) + 1
    
    for reason, count in decisions_by_type.items():
        print(f"  {reason}: {count} times")
    
    return maze, result_1hop, result_2hop


def visualize_large_maze_results(maze: np.ndarray, result_1hop: Dict, result_2hop: Dict):
    """Visualize large maze navigation results"""
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Maze only
    axes[0].imshow(maze, cmap='binary', interpolation='nearest')
    axes[0].set_title("50x50 Complex Maze")
    axes[0].axis('off')
    
    # 1-hop result
    axes[1].imshow(maze, cmap='binary', interpolation='nearest')
    if result_1hop['path'] and len(result_1hop['path']) > 0:
        # Show visited cells
        visited_mask = np.zeros_like(maze, dtype=float)
        for v in result_1hop.get('visited', []):
            if 0 <= v[0] < 50 and 0 <= v[1] < 50:
                visited_mask[v[0], v[1]] = 0.5
        axes[1].imshow(visited_mask, cmap='Reds', alpha=0.5, interpolation='nearest')
        
        # Show path
        if len(result_1hop['path']) > 1:
            path = np.array(result_1hop['path'])
            axes[1].plot(path[:, 1], path[:, 0], 'b-', linewidth=2, alpha=0.7)
            axes[1].plot(path[0, 1], path[0, 0], 'go', markersize=10)
            axes[1].plot(path[-1, 1], path[-1, 0], 'ro', markersize=10)
    
    axes[1].set_title(f"1-hop: {'Success' if result_1hop['success'] else 'Failed'} "
                     f"({result_1hop['steps']} steps)")
    axes[1].axis('off')
    
    # 2-hop result
    axes[2].imshow(maze, cmap='binary', interpolation='nearest')
    if result_2hop['path'] and len(result_2hop['path']) > 0:
        # Show dead-ends detected
        dead_end_mask = np.zeros_like(maze, dtype=float)
        for d in result_2hop.get('dead_ends_found', []):
            if isinstance(d, tuple) and len(d) == 2:
                if 0 <= d[0] < 50 and 0 <= d[1] < 50:
                    dead_end_mask[d[0], d[1]] = 1.0
        if np.any(dead_end_mask > 0):
            axes[2].imshow(dead_end_mask, cmap='Oranges', alpha=0.5, interpolation='nearest')
        
        # Show path
        if len(result_2hop['path']) > 1:
            path = np.array(result_2hop['path'])
            axes[2].plot(path[:, 1], path[:, 0], 'g-', linewidth=2, alpha=0.7)
            axes[2].plot(path[0, 1], path[0, 0], 'go', markersize=10)
            axes[2].plot(path[-1, 1], path[-1, 0], 'ro', markersize=10)
    
    axes[2].set_title(f"2-hop: {'Success' if result_2hop['success'] else 'Failed'} "
                     f"({result_2hop['steps']} steps)")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'large_maze_2hop_comparison_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {filename}")


def run_multiple_trials(n_trials: int = 5):
    """Run multiple trials to get statistics"""
    
    print("Running multiple trials...")
    
    results = {
        '1hop': [],
        '2hop': []
    }
    
    for trial in range(n_trials):
        print(f"\n{'='*50}")
        print(f"TRIAL {trial + 1}/{n_trials}")
        print(f"{'='*50}")
        
        # Generate new maze for each trial
        maze = generate_complex_maze(50, 
                                   complexity=0.3 + 0.2 * np.random.random(),
                                   density=0.3 + 0.1 * np.random.random())
        
        start = (1, 1)
        goal = (48, 48)
        
        # 1-hop
        nav_1hop = LargeMaze2HopNavigator(maze, sparse_graph=True)
        result_1hop = nav_1hop.navigate(start, goal, use_2hop=False, max_steps=3000)
        results['1hop'].append(result_1hop)
        
        # 2-hop
        nav_2hop = LargeMaze2HopNavigator(maze, sparse_graph=True)
        result_2hop = nav_2hop.navigate(start, goal, use_2hop=True, max_steps=3000)
        results['2hop'].append(result_2hop)
        
        print(f"\nTrial {trial + 1} Summary:")
        print(f"  1-hop: {'Success' if result_1hop['success'] else 'Failed'} in {result_1hop['steps']} steps")
        print(f"  2-hop: {'Success' if result_2hop['success'] else 'Failed'} in {result_2hop['steps']} steps")
    
    # Aggregate statistics
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")
    
    for method in ['1hop', '2hop']:
        successes = sum(r['success'] for r in results[method])
        success_rate = successes / n_trials * 100
        
        successful_results = [r for r in results[method] if r['success']]
        
        if successful_results:
            avg_steps = np.mean([r['steps'] for r in successful_results])
            avg_backtracks = np.mean([r['backtrack_count'] for r in successful_results])
            avg_time = np.mean([r['elapsed_time'] for r in successful_results])
        else:
            avg_steps = avg_backtracks = avg_time = 0
        
        print(f"\n{method.upper()}:")
        print(f"  Success rate: {success_rate:.1f}% ({successes}/{n_trials})")
        if successful_results:
            print(f"  Avg steps (successful): {avg_steps:.1f}")
            print(f"  Avg backtracks: {avg_backtracks:.1f}")
            print(f"  Avg time: {avg_time:.2f}s")


if __name__ == "__main__":
    print("Large Maze 2-hop Evaluation Experiment")
    print("=" * 70)
    
    # Single detailed experiment
    maze, result_1hop, result_2hop = run_large_maze_experiment()
    
    # Multiple trials for statistics
    print("\n\nRunning statistical comparison...")
    run_multiple_trials(5)
    
    print("\nExperiment complete!")