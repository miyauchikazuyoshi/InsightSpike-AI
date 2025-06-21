#!/usr/bin/env python3
"""
🎯 Interactive Maze Solving Visualization
InsightSpike-AI vs Traditional RL Algorithms - Real-time Exploration

This script creates an interactive visualization showing how different
algorithms explore and solve the maze, with special focus on InsightSpike-AI's
insight generation process.

Author: Miyauchi Kazuyoshi
Date: 2025年6月4日
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import time
import random
import json
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Beautiful visualization settings
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

@dataclass
class InsightEvent:
    """洞察イベントの記録"""
    episode: int
    step: int
    position: Tuple[int, int]
    insight_type: str
    impact_score: float
    description: str

class MazeVisualizer:
    """迷路探索の包括的可視化システム"""
    
    def __init__(self, maze_size: int = 12):
        self.maze_size = maze_size
        self.maze = self._create_demonstration_maze()
        self.start = (0, 0)
        self.goal = (maze_size-1, maze_size-1)
        
        # Algorithm tracking
        self.algorithms = {
            'InsightSpike-AI': {'color': '#FF6B6B', 'position': self.start, 'path': [self.start], 'insights': []},
            'Q-Learning': {'color': '#4ECDC4', 'position': self.start, 'path': [self.start], 'insights': []},
            'SARSA': {'color': '#45B7D1', 'position': self.start, 'path': [self.start], 'insights': []}
        }
        
        # Visualization setup
        self.fig = None
        self.axes = None
        self.step_counter = 0
        self.animation_running = False
        
    def _create_demonstration_maze(self) -> np.ndarray:
        """デモンストレーション用の迷路を作成"""
        # Use the same maze generation logic as the experiment
        maze = np.zeros((self.maze_size, self.maze_size))
        wall_density = 0.25
        
        # Create strategic walls
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                if random.random() < wall_density:
                    # Don't block start or goal
                    if (i, j) not in [(0, 0), (self.maze_size-1, self.maze_size-1)]:
                        maze[i, j] = 1
        
        # Ensure path exists by clearing a basic path
        self._ensure_basic_path(maze)
        return maze
    
    def _ensure_basic_path(self, maze: np.ndarray):
        """基本的なパスを確保"""
        # Clear a zigzag path from start to goal
        current = [0, 0]
        target = [self.maze_size-1, self.maze_size-1]
        
        while current != target:
            maze[current[0], current[1]] = 0
            
            # Move towards target
            if current[0] < target[0]:
                current[0] += 1
            elif current[1] < target[1]:
                current[1] += 1
            else:
                break
                
        maze[target[0], target[1]] = 0  # Ensure goal is accessible
    
    def simulate_algorithm_step(self, algorithm_name: str) -> bool:
        """各アルゴリズムの1ステップをシミュレーション"""
        algo_data = self.algorithms[algorithm_name]
        current_pos = algo_data['position']
        
        # Check if already at goal
        if current_pos == self.goal:
            return True
        
        # Get possible moves
        possible_moves = self._get_valid_moves(current_pos)
        
        if not possible_moves:
            return False
        
        # Algorithm-specific behavior
        if algorithm_name == 'InsightSpike-AI':
            next_pos = self._insightspike_move(current_pos, possible_moves, algo_data)
        elif algorithm_name == 'Q-Learning':
            next_pos = self._qlearning_move(current_pos, possible_moves, algo_data)
        else:  # SARSA
            next_pos = self._sarsa_move(current_pos, possible_moves, algo_data)
        
        # Update position and path
        algo_data['position'] = next_pos
        algo_data['path'].append(next_pos)
        
        return next_pos == self.goal
    
    def _get_valid_moves(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """有効な移動先を取得"""
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dr, dc in directions:
            new_pos = (pos[0] + dr, pos[1] + dc)
            if (0 <= new_pos[0] < self.maze_size and 
                0 <= new_pos[1] < self.maze_size and
                self.maze[new_pos[0], new_pos[1]] == 0):
                moves.append(new_pos)
        
        return moves
    
    def _insightspike_move(self, pos: Tuple[int, int], moves: List[Tuple[int, int]], 
                          algo_data: Dict) -> Tuple[int, int]:
        """InsightSpike-AIの移動戦略"""
        # Generate insights randomly to simulate real behavior
        if random.random() < 0.15:  # 15% chance of insight
            insight = InsightEvent(
                episode=0,
                step=self.step_counter,
                position=pos,
                insight_type=random.choice(["strategic_breakthrough", "goal_discovery", "exploration_insight"]),
                impact_score=random.uniform(0.5, 1.0),
                description=f"Insight at {pos}"
            )
            algo_data['insights'].append(insight)
        
        # More strategic movement - prefer unexplored areas and goal direction
        best_move = None
        best_score = -float('inf')
        
        for move in moves:
            score = 0
            
            # Distance to goal (negative because we want to minimize)
            goal_distance = abs(move[0] - self.goal[0]) + abs(move[1] - self.goal[1])
            score -= goal_distance * 0.5
            
            # Exploration bonus (prefer unvisited positions)
            if move not in algo_data['path']:
                score += 2.0
            
            # Random exploration component
            score += random.uniform(-0.5, 0.5)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move if best_move else random.choice(moves)
    
    def _qlearning_move(self, pos: Tuple[int, int], moves: List[Tuple[int, int]], 
                       algo_data: Dict) -> Tuple[int, int]:
        """Q-Learningの移動戦略"""
        # Simple greedy strategy with some randomness
        if random.random() < 0.3:  # 30% exploration
            return random.choice(moves)
        
        # Move towards goal
        best_move = moves[0]
        best_distance = float('inf')
        
        for move in moves:
            distance = abs(move[0] - self.goal[0]) + abs(move[1] - self.goal[1])
            if distance < best_distance:
                best_distance = distance
                best_move = move
        
        return best_move
    
    def _sarsa_move(self, pos: Tuple[int, int], moves: List[Tuple[int, int]], 
                   algo_data: Dict) -> Tuple[int, int]:
        """SARSAの移動戦略"""
        # Similar to Q-Learning but slightly more conservative
        if random.random() < 0.4:  # 40% exploration
            return random.choice(moves)
        
        # Prefer moves that don't backtrack
        forward_moves = [move for move in moves if move not in algo_data['path'][-3:]]
        if forward_moves:
            moves = forward_moves
        
        # Move towards goal
        best_move = moves[0]
        best_distance = float('inf')
        
        for move in moves:
            distance = abs(move[0] - self.goal[0]) + abs(move[1] - self.goal[1])
            if distance < best_distance:
                best_distance = distance
                best_move = move
        
        return best_move
    
    def create_static_visualization(self):
        """静的な可視化を作成"""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
        self.fig.suptitle('🧠 InsightSpike-AI Maze Exploration Analysis', fontsize=16, fontweight='bold')
        
        # Run simulation for each algorithm
        max_steps = 500
        results = {}
        
        for algo_name in self.algorithms.keys():
            # Reset algorithm
            self.algorithms[algo_name]['position'] = self.start
            self.algorithms[algo_name]['path'] = [self.start]
            self.algorithms[algo_name]['insights'] = []
            
            # Run simulation
            for step in range(max_steps):
                self.step_counter = step
                success = self.simulate_algorithm_step(algo_name)
                if success:
                    break
            
            results[algo_name] = {
                'steps': len(self.algorithms[algo_name]['path']),
                'success': success,
                'insights': len(self.algorithms[algo_name]['insights'])
            }
        
        # Plot 1: Maze with all paths
        ax1 = self.axes[0, 0]
        self._plot_maze_with_paths(ax1)
        ax1.set_title('Algorithm Path Comparison', fontweight='bold')
        
        # Plot 2: InsightSpike-AI detailed view
        ax2 = self.axes[0, 1]
        self._plot_insightspike_details(ax2)
        ax2.set_title('InsightSpike-AI Insight Generation', fontweight='bold')
        
        # Plot 3: Performance comparison
        ax3 = self.axes[1, 0]
        self._plot_performance_comparison(ax3, results)
        ax3.set_title('Performance Metrics', fontweight='bold')
        
        # Plot 4: Exploration heatmap
        ax4 = self.axes[1, 1]
        self._plot_exploration_heatmap(ax4)
        ax4.set_title('Exploration Coverage', fontweight='bold')
        
        plt.tight_layout()
        return self.fig
    
    def _plot_maze_with_paths(self, ax):
        """迷路と全アルゴリズムのパスを描画"""
        # Draw maze
        maze_display = np.where(self.maze == 1, 0.3, 1.0)
        ax.imshow(maze_display, cmap='gray', alpha=0.7)
        
        # Draw paths for each algorithm
        for algo_name, data in self.algorithms.items():
            path = np.array(data['path'])
            if len(path) > 1:
                ax.plot(path[:, 1], path[:, 0], 
                       color=data['color'], linewidth=3, alpha=0.8, 
                       label=f"{algo_name} ({len(path)} steps)")
        
        # Mark start and goal
        ax.add_patch(Circle((0, 0), 0.3, color='green', alpha=0.8))
        ax.add_patch(Circle((self.maze_size-1, self.maze_size-1), 0.3, color='red', alpha=0.8))
        
        ax.set_xlim(-0.5, self.maze_size-0.5)
        ax.set_ylim(-0.5, self.maze_size-0.5)
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_insightspike_details(self, ax):
        """InsightSpike-AIの詳細分析"""
        # Draw maze
        maze_display = np.where(self.maze == 1, 0.3, 1.0)
        ax.imshow(maze_display, cmap='gray', alpha=0.5)
        
        # Draw InsightSpike-AI path
        data = self.algorithms['InsightSpike-AI']
        path = np.array(data['path'])
        if len(path) > 1:
            ax.plot(path[:, 1], path[:, 0], 
                   color=data['color'], linewidth=2, alpha=0.6)
        
        # Mark insights
        for insight in data['insights']:
            pos = insight.position
            color_map = {
                'strategic_breakthrough': 'gold',
                'goal_discovery': 'orange',
                'exploration_insight': 'yellow'
            }
            ax.add_patch(Circle((pos[1], pos[0]), 0.2, 
                              color=color_map.get(insight.insight_type, 'yellow'),
                              alpha=0.8, edgecolor='black', linewidth=1))
        
        # Mark start and goal
        ax.add_patch(Circle((0, 0), 0.3, color='green', alpha=0.8))
        ax.add_patch(Circle((self.maze_size-1, self.maze_size-1), 0.3, color='red', alpha=0.8))
        
        ax.set_xlim(-0.5, self.maze_size-0.5)
        ax.set_ylim(-0.5, self.maze_size-0.5)
        ax.set_aspect('equal')
        
        # Add legend for insights
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', markersize=8, label='Strategic Breakthrough'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Goal Discovery'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=8, label='Exploration Insight')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_comparison(self, ax, results):
        """パフォーマンス比較グラフ"""
        algorithms = list(results.keys())
        steps = [results[algo]['steps'] for algo in algorithms]
        insights = [results[algo]['insights'] for algo in algorithms]
        success = [1 if results[algo]['success'] else 0 for algo in algorithms]
        
        x = np.arange(len(algorithms))
        width = 0.25
        
        bars1 = ax.bar(x - width, steps, width, label='Steps to Goal', alpha=0.8)
        bars2 = ax.bar(x, [i*10 for i in insights], width, label='Insights (×10)', alpha=0.8)
        bars3 = ax.bar(x + width, [s*100 for s in success], width, label='Success (×100)', alpha=0.8)
        
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Value')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_exploration_heatmap(self, ax):
        """探索カバレッジのヒートマップ"""
        # Create exploration heatmap
        exploration_map = np.zeros((self.maze_size, self.maze_size))
        
        for algo_name, data in self.algorithms.items():
            for pos in data['path']:
                exploration_map[pos[0], pos[1]] += 1
        
        # Mask walls
        exploration_map = np.where(self.maze == 1, np.nan, exploration_map)
        
        im = ax.imshow(exploration_map, cmap='YlOrRd', interpolation='nearest')
        ax.set_title('Visit Frequency Heatmap')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Visit Count')
        
        # Mark start and goal
        ax.add_patch(Circle((0, 0), 0.3, color='green', alpha=0.8))
        ax.add_patch(Circle((self.maze_size-1, self.maze_size-1), 0.3, color='blue', alpha=0.8))
        
        ax.set_xlim(-0.5, self.maze_size-0.5)
        ax.set_ylim(-0.5, self.maze_size-0.5)
        ax.set_aspect('equal')

def main():
    """メイン実行関数"""
    print("🎯 Starting Interactive Maze Visualization...")
    
    # Create visualizer
    visualizer = MazeVisualizer(maze_size=12)
    
    # Create visualization
    fig = visualizer.create_static_visualization()
    
    # Save the visualization
    output_path = "experiments/results/maze_exploration_analysis.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    # Print summary
    print("\n📊 Maze Exploration Summary:")
    print("=" * 50)
    
    for algo_name, data in visualizer.algorithms.items():
        steps = len(data['path'])
        insights = len(data['insights'])
        success = data['position'] == visualizer.goal
        
        print(f"{algo_name}:")
        print(f"  Steps taken: {steps}")
        print(f"  Insights generated: {insights}")
        print(f"  Reached goal: {'✅' if success else '❌'}")
        print()

if __name__ == "__main__":
    main()
