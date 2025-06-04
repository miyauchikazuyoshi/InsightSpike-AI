#!/usr/bin/env python3
"""
🎬 InsightSpike-AI 動的迷路探索アニメーション
Dynamic Real-Time Maze Exploration Animation

InsightSpike-AIの迷路探索プロセスをリアルタイムアニメーションで可視化し、
洞察生成の瞬間を視覚的に捉えます。

Author: Miyauchi Kazuyoshi
Date: 2025年6月4日
特許出願中: JP特願2025-082988, JP特願2025-082989
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
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
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 12
sns.set_style("white")

@dataclass
class InsightMoment:
    """革新的洞察瞬間の記録"""
    episode: int
    step: int
    dged_value: float
    dig_value: float
    state: Tuple[int, int]
    action: str
    insight_type: str
    description: str
    performance_impact: float

class AnimatedMazeEnvironment:
    """アニメーション用迷路環境"""
    
    def __init__(self, size: int = 10, wall_density: float = 0.2):
        self.size = size
        self.wall_density = wall_density
        self.reset()
        
    def reset(self):
        """環境リセット"""
        # Create simple maze for animation
        self.maze = np.zeros((self.size, self.size))
        
        # Add strategic walls
        for i in range(1, self.size-1):
            for j in range(1, self.size-1):
                if random.random() < self.wall_density:
                    self.maze[i, j] = 1
        
        # Ensure borders are walls
        self.maze[0, :] = 1
        self.maze[-1, :] = 1
        self.maze[:, 0] = 1
        self.maze[:, -1] = 1
        
        # Set start and goal
        self.start = (1, 1)
        self.goal = (self.size-2, self.size-2)
        self.maze[self.start] = 0
        self.maze[self.goal] = 0
        
        # Create guaranteed path
        self._create_path()
        
        # Initialize state
        self.current_pos = self.start
        self.visited_states = {self.start}
        self.step_count = 0
        
        return self.current_pos
    
    def _create_path(self):
        """確実なパスを作成"""
        # Simple path creation
        r, c = self.start
        goal_r, goal_c = self.goal
        
        # Create L-shaped path
        while r < goal_r:
            self.maze[r, c] = 0
            r += 1
        while c < goal_c:
            self.maze[r, c] = 0
            c += 1
        self.maze[goal_r, goal_c] = 0

class AnimatedInsightAgent:
    """アニメーション用InsightSpike-AIエージェント"""
    
    def __init__(self, env: AnimatedMazeEnvironment):
        self.env = env
        self.q_table = defaultdict(lambda: np.zeros(4))
        self.epsilon = 0.3
        self.learning_rate = 0.1
        
        # Insight detection
        self.insights = []
        self.exploration_memory = defaultdict(int)
        self.recent_insights = deque(maxlen=5)  # For animation
        
    def calculate_insight_probability(self, state: Tuple[int, int]) -> float:
        """洞察確率の計算"""
        # Distance to goal
        goal_distance = abs(state[0] - self.env.goal[0]) + abs(state[1] - self.env.goal[1])
        goal_factor = 1.0 / (1 + goal_distance)
        
        # Exploration factor
        visit_count = self.exploration_memory[state]
        exploration_factor = 1.0 / (1 + visit_count)
        
        # Neighbor exploration
        unexplored_neighbors = 0
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = state[0] + dr, state[1] + dc
            if (0 <= nr < self.env.size and 0 <= nc < self.env.size and
                (nr, nc) not in self.env.visited_states):
                unexplored_neighbors += 1
        
        neighbor_factor = unexplored_neighbors / 4.0
        
        return goal_factor * 0.4 + exploration_factor * 0.3 + neighbor_factor * 0.3
    
    def choose_action(self, state: Tuple[int, int]) -> Tuple[int, bool]:
        """アクション選択と洞察検出"""
        # Generate insight
        insight_detected = False
        insight_prob = self.calculate_insight_probability(state)
        
        if random.random() < insight_prob * 0.3:  # Scale down for animation
            insight_detected = True
            insight = InsightMoment(
                episode=0,
                step=self.env.step_count,
                dged_value=insight_prob,
                dig_value=insight_prob,
                state=state,
                action="explore",
                insight_type="exploration_insight",
                description=f"Strategic insight at {state}",
                performance_impact=insight_prob
            )
            self.insights.append(insight)
            self.recent_insights.append(insight)
        
        # Choose action
        if random.random() < self.epsilon:
            # Intelligent exploration
            valid_actions = []
            for action in range(4):
                moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                dr, dc = moves[action]
                next_pos = (state[0] + dr, state[1] + dc)
                
                if (0 <= next_pos[0] < self.env.size and 
                    0 <= next_pos[1] < self.env.size and
                    self.env.maze[next_pos] == 0):
                    valid_actions.append(action)
            
            if valid_actions:
                action = random.choice(valid_actions)
            else:
                action = random.randint(0, 3)
        else:
            action = np.argmax(self.q_table[state])
        
        return action, insight_detected

class RealTimeMazeAnimation:
    """リアルタイム迷路アニメーション"""
    
    def __init__(self):
        self.env = AnimatedMazeEnvironment()
        self.agent = AnimatedInsightAgent(self.env)
        
        # Animation setup
        self.fig, (self.ax_maze, self.ax_insights) = plt.subplots(1, 2, figsize=(16, 8))
        self.fig.suptitle('🧠 InsightSpike-AI: リアルタイム迷路探索', fontsize=16, fontweight='bold')
        
        # Animation data
        self.path_history = []
        self.insight_history = []
        self.current_episode = 0
        self.max_steps = 200
        self.step_count = 0
        
        # Visual elements
        self.agent_marker = None
        self.insight_burst = None
        self.path_line = None
        
    def setup_maze_plot(self):
        """迷路プロットの初期設定"""
        self.ax_maze.clear()
        self.ax_maze.set_title('🗺️ 迷路探索 & 洞察生成', fontsize=14, fontweight='bold')
        
        # Draw maze
        maze_display = self.env.maze.copy()
        self.ax_maze.imshow(maze_display, cmap='binary', alpha=0.7)
        
        # Mark special positions
        self.ax_maze.scatter(self.env.start[1], self.env.start[0], 
                           c='green', s=300, marker='o', label='Start', 
                           edgecolors='black', linewidth=2)
        self.ax_maze.scatter(self.env.goal[1], self.env.goal[0], 
                           c='red', s=400, marker='*', label='Goal',
                           edgecolors='black', linewidth=2)
        
        # Agent marker (initial)
        self.agent_marker = self.ax_maze.scatter(self.env.current_pos[1], self.env.current_pos[0], 
                                               c='blue', s=200, marker='o', 
                                               edgecolors='white', linewidth=3, zorder=10)
        
        self.ax_maze.set_xlim(-0.5, self.env.size-0.5)
        self.ax_maze.set_ylim(-0.5, self.env.size-0.5)
        self.ax_maze.set_xticks([])
        self.ax_maze.set_yticks([])
        self.ax_maze.legend(loc='upper right')
        
    def setup_insight_plot(self):
        """洞察プロットの初期設定"""
        self.ax_insights.clear()
        self.ax_insights.set_title('💡 洞察生成ダッシュボード', fontsize=14, fontweight='bold')
        self.ax_insights.set_xlim(0, 100)
        self.ax_insights.set_ylim(0, 10)
        self.ax_insights.set_xlabel('ステップ')
        self.ax_insights.set_ylabel('洞察強度')
        
    def animate_step(self, frame):
        """アニメーションの1ステップ"""
        if self.step_count >= self.max_steps:
            return
        
        current_state = self.env.current_pos
        
        # Agent chooses action
        action, insight_detected = self.agent.choose_action(current_state)
        
        # Execute action
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        dr, dc = moves[action]
        next_state = (current_state[0] + dr, current_state[1] + dc)
        
        # Check if move is valid
        if (0 <= next_state[0] < self.env.size and 
            0 <= next_state[1] < self.env.size and
            self.env.maze[next_state] == 0):
            
            # Move agent
            self.env.current_pos = next_state
            self.env.visited_states.add(next_state)
            self.agent.exploration_memory[next_state] += 1
            self.path_history.append(next_state)
            
            # Update agent position on plot
            self.agent_marker.set_offsets([[next_state[1], next_state[0]]])
            
        # Draw path
        if len(self.path_history) > 1:
            path_x = [pos[1] for pos in self.path_history]
            path_y = [pos[0] for pos in self.path_history]
            
            # Clear previous path and redraw
            for line in self.ax_maze.lines:
                line.remove()
            
            self.ax_maze.plot(path_x, path_y, 'b-', alpha=0.6, linewidth=2, zorder=5)
        
        # Draw visited states
        if len(self.path_history) > 0:
            visited_x = [pos[1] for pos in self.env.visited_states]
            visited_y = [pos[0] for pos in self.env.visited_states]
            self.ax_maze.scatter(visited_x, visited_y, c='lightblue', s=50, alpha=0.6, zorder=3)
        
        # Insight visualization
        if insight_detected:
            # Create insight burst effect
            insight_x, insight_y = self.env.current_pos[1], self.env.current_pos[0]
            
            # Add insight burst
            circle = Circle((insight_x, insight_y), 0.8, color='yellow', alpha=0.7, zorder=8)
            self.ax_maze.add_patch(circle)
            
            # Update insight plot
            self.insight_history.append((self.step_count, self.agent.recent_insights[-1].performance_impact))
        
        # Update insight dashboard
        if self.insight_history:
            steps, impacts = zip(*self.insight_history)
            self.ax_insights.clear()
            self.ax_insights.set_title('💡 洞察生成ダッシュボード', fontsize=14, fontweight='bold')
            self.ax_insights.plot(steps, impacts, 'r-', linewidth=2, marker='o', markersize=8)
            self.ax_insights.scatter(steps, impacts, c='red', s=100, alpha=0.8, zorder=10)
            self.ax_insights.set_xlim(max(0, self.step_count-50), self.step_count+10)
            self.ax_insights.set_ylim(0, 1)
            self.ax_insights.set_xlabel('ステップ')
            self.ax_insights.set_ylabel('洞察強度')
            self.ax_insights.grid(True, alpha=0.3)
            
            # Add insight counter
            self.ax_insights.text(0.02, 0.98, f'総洞察数: {len(self.agent.insights)}', 
                                transform=self.ax_insights.transAxes, 
                                fontsize=12, fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Update step counter
        self.step_count += 1
        self.env.step_count += 1
        
        # Add step counter to maze plot
        self.ax_maze.text(0.02, 0.98, f'ステップ: {self.step_count}', 
                         transform=self.ax_maze.transAxes, 
                         fontsize=12, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # Check if goal reached
        if self.env.current_pos == self.env.goal:
            self.ax_maze.text(0.5, 0.02, '🎯 GOAL REACHED!', 
                             transform=self.ax_maze.transAxes, 
                             fontsize=16, fontweight='bold', ha='center',
                             bbox=dict(boxstyle="round,pad=0.5", facecolor="gold", alpha=0.9))
    
    def run_animation(self):
        """アニメーション実行"""
        print("🎬 リアルタイム迷路探索アニメーション開始...")
        print(f"🗺️ 迷路サイズ: {self.env.size}x{self.env.size}")
        print(f"📍 Start: {self.env.start}, Goal: {self.env.goal}")
        print("=" * 50)
        
        # Setup plots
        self.setup_maze_plot()
        self.setup_insight_plot()
        
        # Create animation
        anim = animation.FuncAnimation(
            self.fig, self.animate_step, 
            frames=self.max_steps, 
            interval=200,  # 200ms between frames
            repeat=False,
            blit=False
        )
        
        plt.tight_layout()
        plt.show()
        
        # Print final results
        print("\n" + "=" * 50)
        print("🎯 アニメーション完了")
        print(f"🧠 総洞察数: {len(self.agent.insights)}")
        print(f"👣 総ステップ数: {self.step_count}")
        print(f"🎯 ゴール到達: {'Yes' if self.env.current_pos == self.env.goal else 'No'}")
        
        return anim

def main():
    """メイン実行関数"""
    print("🧠 InsightSpike-AI リアルタイム迷路探索アニメーション")
    print("=" * 60)
    
    # Create and run animation
    animation_system = RealTimeMazeAnimation()
    anim = animation_system.run_animation()
    
    # Optionally save animation as GIF
    save_animation = input("\n💾 アニメーションをGIFとして保存しますか？ (y/n): ").lower() == 'y'
    
    if save_animation:
        print("🎬 GIF生成中...")
        results_dir = "/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/experiments/results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = f"{results_dir}/maze_exploration_animation_{timestamp}.gif"
        
        try:
            anim.save(gif_path, writer='pillow', fps=5, dpi=100)
            print(f"✅ GIF保存完了: {gif_path}")
        except Exception as e:
            print(f"❌ GIF保存エラー: {e}")

if __name__ == "__main__":
    main()
