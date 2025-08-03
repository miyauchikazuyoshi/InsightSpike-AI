#!/usr/bin/env python3
"""
Test 2-hop evaluation in maze navigation
=========================================

Verify if 2-hop structural evaluation can detect dead-ends early
and improve backtracking decisions.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
import time


class Maze2HopNavigator:
    """Maze navigator with 2-hop structural evaluation"""
    
    def __init__(self, maze: np.ndarray):
        self.maze = maze
        self.height, self.width = maze.shape
        self.visited = set()
        self.path = []
        self.decision_history = []
        
        # Build maze graph for structural analysis
        self.maze_graph = self._build_maze_graph()
        
    def _build_maze_graph(self) -> nx.Graph:
        """Build graph representation of maze paths"""
        G = nx.Graph()
        
        # Add nodes for all walkable cells
        for i in range(self.height):
            for j in range(self.width):
                if self.maze[i, j] == 0:  # Walkable
                    G.add_node((i, j))
        
        # Add edges between adjacent walkable cells
        for i in range(self.height):
            for j in range(self.width):
                if self.maze[i, j] == 0:
                    # Check 4 directions
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < self.height and 
                            0 <= nj < self.width and 
                            self.maze[ni, nj] == 0):
                            G.add_edge((i, j), (ni, nj))
        
        return G
    
    def evaluate_1hop(self, position: Tuple[int, int], 
                      goal: Tuple[int, int]) -> Dict:
        """Standard 1-hop evaluation (distance-based)"""
        neighbors = self._get_neighbors(position)
        
        best_action = None
        best_distance = float('inf')
        
        for neighbor in neighbors:
            if neighbor not in self.visited:
                dist = np.sqrt((neighbor[0] - goal[0])**2 + 
                              (neighbor[1] - goal[1])**2)
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
        """2-hop structural evaluation for dead-end detection"""
        
        # Extract 2-hop subgraph
        subgraph = self._extract_2hop_subgraph(position)
        
        # Calculate structural metrics
        metrics = self._calculate_structural_metrics(subgraph, position, goal)
        
        # Detect if current path leads to dead-end
        if self._is_structural_deadend(metrics):
            # Find backtrack point
            backtrack_point = self._find_backtrack_point(position)
            return {
                'action': 'backtrack',
                'target': backtrack_point,
                'reason': 'structural_deadend',
                'metrics': metrics,
                'type': '2hop'
            }
        
        # Otherwise, use 1-hop decision
        return self.evaluate_1hop(position, goal)
    
    def _extract_2hop_subgraph(self, center: Tuple[int, int]) -> nx.Graph:
        """Extract 2-hop neighborhood subgraph"""
        if center not in self.maze_graph:
            return nx.Graph()
        
        # BFS to find 2-hop neighbors
        nodes = {center}
        for _ in range(2):
            new_nodes = set()
            for node in nodes:
                if node in self.maze_graph:
                    new_nodes.update(self.maze_graph.neighbors(node))
            nodes.update(new_nodes)
        
        # Create subgraph
        return self.maze_graph.subgraph(nodes)
    
    def _calculate_structural_metrics(self, subgraph: nx.Graph, 
                                    position: Tuple[int, int],
                                    goal: Tuple[int, int]) -> Dict:
        """Calculate structural metrics for dead-end detection"""
        
        metrics = {
            'node_count': subgraph.number_of_nodes(),
            'edge_count': subgraph.number_of_edges(),
            'avg_degree': 0,
            'has_goal': goal in subgraph.nodes,
            'bottlenecks': 0,
            'dead_end_probability': 0
        }
        
        if subgraph.number_of_nodes() == 0:
            return metrics
        
        # Average degree (connectivity)
        degrees = dict(subgraph.degree())
        metrics['avg_degree'] = np.mean(list(degrees.values()))
        
        # Count bottlenecks (degree 1 nodes)
        metrics['bottlenecks'] = sum(1 for d in degrees.values() if d == 1)
        
        # Dead-end probability
        if position in degrees:
            # Low connectivity + many bottlenecks = likely dead-end
            metrics['dead_end_probability'] = (
                metrics['bottlenecks'] / max(1, metrics['node_count']) *
                (2.0 / max(1, metrics['avg_degree']))
            )
        
        return metrics
    
    def _is_structural_deadend(self, metrics: Dict) -> bool:
        """Determine if current path is a dead-end based on structure"""
        return (
            metrics['dead_end_probability'] > 0.7 and
            not metrics['has_goal'] and
            metrics['avg_degree'] < 2.5
        )
    
    def _find_backtrack_point(self, position: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find junction point to backtrack to"""
        # Look for recent junction in path
        for i in range(len(self.path) - 1, -1, -1):
            pos = self.path[i]
            if self._count_unvisited_neighbors(pos) > 1:
                return pos
        return None
    
    def _count_unvisited_neighbors(self, position: Tuple[int, int]) -> int:
        """Count unvisited neighbors of a position"""
        count = 0
        for neighbor in self._get_neighbors(position):
            if neighbor not in self.visited:
                count += 1
        return count
    
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
    
    def navigate(self, start: Tuple[int, int], goal: Tuple[int, int],
                use_2hop: bool = True) -> Dict:
        """Navigate maze with optional 2-hop evaluation"""
        
        self.visited = {start}
        self.path = [start]
        self.decision_history = []
        position = start
        
        steps = 0
        backtrack_count = 0
        
        while position != goal and steps < 1000:
            # Evaluate next action
            if use_2hop:
                decision = self.evaluate_2hop(position, goal)
            else:
                decision = self.evaluate_1hop(position, goal)
            
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
                    # Backtrack to target
                    idx = self.path.index(target)
                    self.path = self.path[:idx + 1]
                    position = target
                else:
                    # No valid backtrack, try 1-hop
                    decision = self.evaluate_1hop(position, goal)
                    if decision['action']:
                        position = decision['action']
                        self.visited.add(position)
                        self.path.append(position)
                    else:
                        break  # Stuck
            else:
                # Normal movement
                if decision['action']:
                    position = decision['action']
                    self.visited.add(position)
                    self.path.append(position)
                else:
                    # Dead end, backtrack manually
                    backtrack_count += 1
                    if len(self.path) > 1:
                        self.path.pop()
                        position = self.path[-1]
                    else:
                        break  # Stuck at start
            
            steps += 1
        
        success = position == goal
        
        return {
            'success': success,
            'steps': steps,
            'path_length': len(self.path),
            'backtrack_count': backtrack_count,
            'visited_count': len(self.visited),
            'path': self.path.copy(),
            'decision_history': self.decision_history
        }


def generate_test_maze(size: int = 15, complexity: float = 0.3) -> np.ndarray:
    """Generate a maze with controllable complexity"""
    maze = np.zeros((size, size))
    
    # Add walls
    for i in range(size):
        for j in range(size):
            if np.random.random() < complexity:
                maze[i, j] = 1
    
    # Ensure start and goal are clear
    maze[0, 0] = 0
    maze[size-1, size-1] = 0
    
    # Ensure path exists (simple corridor)
    for i in range(size):
        if i < size // 2:
            maze[i, 0] = 0
        else:
            maze[size-1, i] = 0
    
    return maze


def run_comparison_experiment(n_trials: int = 20):
    """Compare 1-hop vs 2-hop navigation"""
    
    results_1hop = []
    results_2hop = []
    
    for trial in range(n_trials):
        # Generate random maze
        maze = generate_test_maze(15, complexity=0.3 + 0.2 * np.random.random())
        
        # Test 1-hop
        navigator_1hop = Maze2HopNavigator(maze)
        result_1hop = navigator_1hop.navigate((0, 0), (14, 14), use_2hop=False)
        results_1hop.append(result_1hop)
        
        # Test 2-hop
        navigator_2hop = Maze2HopNavigator(maze)
        result_2hop = navigator_2hop.navigate((0, 0), (14, 14), use_2hop=True)
        results_2hop.append(result_2hop)
        
        print(f"Trial {trial + 1}:")
        print(f"  1-hop: Success={result_1hop['success']}, "
              f"Steps={result_1hop['steps']}, "
              f"Backtracks={result_1hop['backtrack_count']}")
        print(f"  2-hop: Success={result_2hop['success']}, "
              f"Steps={result_2hop['steps']}, "
              f"Backtracks={result_2hop['backtrack_count']}")
    
    # Analyze results
    success_1hop = sum(r['success'] for r in results_1hop)
    success_2hop = sum(r['success'] for r in results_2hop)
    
    avg_steps_1hop = np.mean([r['steps'] for r in results_1hop if r['success']])
    avg_steps_2hop = np.mean([r['steps'] for r in results_2hop if r['success']])
    
    avg_backtracks_1hop = np.mean([r['backtrack_count'] for r in results_1hop])
    avg_backtracks_2hop = np.mean([r['backtrack_count'] for r in results_2hop])
    
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    print(f"Success Rate:")
    print(f"  1-hop: {success_1hop}/{n_trials} ({100*success_1hop/n_trials:.1f}%)")
    print(f"  2-hop: {success_2hop}/{n_trials} ({100*success_2hop/n_trials:.1f}%)")
    print(f"\nAverage Steps (successful only):")
    print(f"  1-hop: {avg_steps_1hop:.1f}")
    print(f"  2-hop: {avg_steps_2hop:.1f}")
    print(f"\nAverage Backtracks:")
    print(f"  1-hop: {avg_backtracks_1hop:.1f}")
    print(f"  2-hop: {avg_backtracks_2hop:.1f}")
    
    return results_1hop, results_2hop


def visualize_decision_history(maze: np.ndarray, result: Dict, title: str):
    """Visualize maze navigation with decision points"""
    
    plt.figure(figsize=(10, 10))
    
    # Show maze
    plt.imshow(maze, cmap='binary')
    
    # Show path
    if result['path']:
        path = np.array(result['path'])
        plt.plot(path[:, 1], path[:, 0], 'b-', linewidth=2, alpha=0.7)
        plt.plot(path[0, 1], path[0, 0], 'go', markersize=15, label='Start')
        plt.plot(path[-1, 1], path[-1, 0], 'ro', markersize=15, label='End')
    
    # Show backtrack decisions
    for decision in result['decision_history']:
        if decision['decision'].get('action') == 'backtrack':
            pos = decision['position']
            plt.plot(pos[1], pos[0], 'rx', markersize=10, markeredgewidth=3)
    
    plt.title(title)
    plt.legend()
    plt.axis('off')
    plt.tight_layout()


if __name__ == "__main__":
    print("Testing 2-hop evaluation in maze navigation...")
    print()
    
    # Run comparison
    results_1hop, results_2hop = run_comparison_experiment(20)
    
    # Visualize example
    print("\nGenerating visualization...")
    maze = generate_test_maze(15, 0.35)
    
    nav_1hop = Maze2HopNavigator(maze)
    result_1hop = nav_1hop.navigate((0, 0), (14, 14), use_2hop=False)
    
    nav_2hop = Maze2HopNavigator(maze)
    result_2hop = nav_2hop.navigate((0, 0), (14, 14), use_2hop=True)
    
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    visualize_decision_history(maze, result_1hop, "1-hop Navigation")
    
    plt.subplot(1, 2, 2)
    visualize_decision_history(maze, result_2hop, "2-hop Navigation (X = backtrack)")
    
    plt.savefig('maze_2hop_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to maze_2hop_comparison.png")
    
    # Analyze structural decisions
    print("\n2-hop Structural Decisions:")
    for i, decision in enumerate(result_2hop['decision_history'][:10]):
        if decision['decision'].get('type') == '2hop' and decision['decision'].get('action') == 'backtrack':
            print(f"Step {i}: Backtrack due to {decision['decision']['reason']}")
            print(f"  Metrics: {decision['decision']['metrics']}")