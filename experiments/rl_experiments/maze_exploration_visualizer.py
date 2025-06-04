#!/usr/bin/env python3
"""
🎭 InsightSpike-AI 迷路探索可視化
Dynamic Maze Exploration Visualization

リアルタイムで迷路探索プロセスと洞察生成を可視化します。
InsightSpike-AIがどのように迷路を学習し、革新的な解決策を見つけるかを視覚的に示します。

Author: Miyauchi Kazuyoshi
Date: 2025年6月4日
特許出願中: JP特願2025-082988, JP特願2025-082989
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
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

@dataclass
class InsightMoment:
    """革新的洞察瞬間の記録"""
    episode: int
    step: int
    dged_value: float      # Δ Global Exploration Difficulty
    dig_value: float       # Δ Information Gain
    state: Tuple[int, int]
    action: str
    insight_type: str
    description: str
    performance_impact: float

class DynamicMazeEnvironment:
    """動的迷路環境"""
    
    def __init__(self, size: int = 12, wall_density: float = 0.25):
        self.size = size
        self.wall_density = wall_density
        self.reset_maze()
        
    def reset_maze(self):
        """迷路を初期化"""
        # Create maze with walls
        self.maze = np.zeros((self.size, self.size))
        
        # Add random walls
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
        
        # Ensure path exists
        self._ensure_path_exists()
        
        # Place treasures and traps
        self.treasure_positions = self._place_treasures()
        self.trap_positions = self._place_traps()
        
        # Initialize state
        self.current_pos = self.start
        self.visited_states = {self.start}
        self.step_count = 0
        
        return self.current_pos
        
    def _ensure_path_exists(self):
        """ゴールまでのパスが存在することを保証"""
        visited = set()
        queue = deque([self.start])
        visited.add(self.start)
        
        while queue:
            r, c = queue.popleft()
            if (r, c) == self.goal:
                return True
                
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.size and 0 <= nc < self.size and
                    self.maze[nr, nc] == 0 and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
                    
        # If no path exists, create one
        self._create_path()
        
    def _create_path(self):
        """強制的にパスを作成"""
        r, c = self.start
        goal_r, goal_c = self.goal
        
        # Create a simple path
        while r != goal_r or c != goal_c:
            self.maze[r, c] = 0
            if r < goal_r:
                r += 1
            elif r > goal_r:
                r -= 1
            elif c < goal_c:
                c += 1
            elif c > goal_c:
                c -= 1
    
    def _place_treasures(self) -> List[Tuple[int, int]]:
        """宝箱の戦略的配置"""
        treasures = []
        num_treasures = max(2, self.size // 4)
        
        for _ in range(num_treasures):
            for _ in range(100):  # Max attempts
                pos = (random.randint(1, self.size-2), random.randint(1, self.size-2))
                if (self.maze[pos[0], pos[1]] == 0 and 
                    pos not in [self.start, self.goal] and
                    pos not in treasures):
                    treasures.append(pos)
                    break
        return treasures
    
    def _place_traps(self) -> List[Tuple[int, int]]:
        """トラップの戦略的配置"""
        traps = []
        num_traps = max(1, self.size // 6)
        
        for _ in range(num_traps):
            for _ in range(100):  # Max attempts
                pos = (random.randint(1, self.size-2), random.randint(1, self.size-2))
                if (self.maze[pos[0], pos[1]] == 0 and 
                    pos not in [self.start, self.goal] and
                    pos not in self.treasure_positions and
                    pos not in traps):
                    traps.append(pos)
                    break
        return traps

class InsightSpikeAgent:
    """InsightSpike-AI エージェント"""
    
    def __init__(self, env: DynamicMazeEnvironment, learning_rate: float = 0.1):
        self.env = env
        self.learning_rate = learning_rate
        self.q_table = defaultdict(lambda: np.zeros(4))  # 4 actions
        self.epsilon = 0.9
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # InsightSpike-AI specific parameters
        self.insights = []
        self.exploration_memory = defaultdict(int)
        self.goal_oriented_memory = defaultdict(float)
        self.dged_threshold = 0.3  # Δ Global Exploration Difficulty threshold
        self.dig_threshold = 0.5   # Δ Information Gain threshold
        
        # Tracking variables
        self.path_history = []
        self.insights_per_episode = []
        
    def calculate_dged(self, state: Tuple[int, int]) -> float:
        """Δ Global Exploration Difficulty計算"""
        unexplored_neighbors = 0
        total_neighbors = 0
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = state[0] + dr, state[1] + dc
            if 0 <= nr < self.env.size and 0 <= nc < self.env.size:
                total_neighbors += 1
                if (nr, nc) not in self.env.visited_states:
                    unexplored_neighbors += 1
        
        if total_neighbors == 0:
            return 0.0
        
        return unexplored_neighbors / total_neighbors
    
    def calculate_dig(self, state: Tuple[int, int], action: int) -> float:
        """Δ Information Gain計算"""
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        dr, dc = moves[action]
        next_state = (state[0] + dr, state[1] + dc)
        
        # Information gain from visiting new states
        if next_state not in self.env.visited_states:
            return 1.0
        
        # Information gain from revisiting with new context
        visit_count = self.exploration_memory[next_state]
        return max(0.1, 1.0 / (1 + visit_count))
    
    def detect_insight(self, state: Tuple[int, int], action: int, episode: int, step: int) -> Optional[InsightMoment]:
        """革新的洞察の検出"""
        dged = self.calculate_dged(state)
        dig = self.calculate_dig(state, action)
        
        insight_detected = False
        insight_type = ""
        description = ""
        performance_impact = 0.0
        
        # Strategic breakthrough detection
        if dged > self.dged_threshold and dig > self.dig_threshold:
            insight_detected = True
            insight_type = "strategic_breakthrough"
            description = f"Revolutionary exploration strategy discovered at {state}"
            performance_impact = 0.8
        
        # Goal discovery insight
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        dr, dc = moves[action]
        next_state = (state[0] + dr, state[1] + dc)
        goal_distance = abs(next_state[0] - self.env.goal[0]) + abs(next_state[1] - self.env.goal[1])
        
        if goal_distance <= 2 and next_state not in self.env.visited_states:
            insight_detected = True
            insight_type = "goal_discovery"
            description = f"Goal proximity insight: Close to target at {next_state}"
            performance_impact = 1.0
        
        # Exploration insight
        if dig > 0.7:
            insight_detected = True
            insight_type = "exploration_insight"
            description = f"High-value exploration opportunity at {state}"
            performance_impact = 0.6
        
        if insight_detected:
            action_names = ["right", "left", "down", "up"]
            return InsightMoment(
                episode=episode,
                step=step,
                dged_value=dged,
                dig_value=dig,
                state=state,
                action=action_names[action],
                insight_type=insight_type,
                description=description,
                performance_impact=performance_impact
            )
        
        return None
    
    def choose_action(self, state: Tuple[int, int], episode: int, step: int) -> int:
        """アクション選択（洞察検出付き）"""
        if random.random() < self.epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(self.q_table[state])
        
        # Detect insights
        insight = self.detect_insight(state, action, episode, step)
        if insight:
            self.insights.append(insight)
        
        return action
    
    def update_q_value(self, state: Tuple[int, int], action: int, reward: float, next_state: Tuple[int, int]):
        """Q値更新"""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + 0.95 * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

class MazeVisualizationSystem:
    """迷路探索可視化システム"""
    
    def __init__(self, env: DynamicMazeEnvironment, agent: InsightSpikeAgent):
        self.env = env
        self.agent = agent
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
        self.fig.suptitle('🧠 InsightSpike-AI: 迷路探索＆洞察生成可視化', fontsize=16, fontweight='bold')
        
        # Animation data
        self.episode_data = []
        self.current_episode = 0
        self.max_episodes = 50
        
    def create_static_visualization(self):
        """静的可視化の作成"""
        # Run episodes to collect data
        for episode in range(self.max_episodes):
            episode_insights = []
            episode_path = []
            
            self.env.reset_maze()
            state = self.env.current_pos
            episode_path.append(state)
            
            for step in range(100):  # Max steps per episode
                action = self.agent.choose_action(state, episode, step)
                
                # Execute action
                moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                dr, dc = moves[action]
                next_state = (state[0] + dr, state[1] + dc)
                
                # Check boundaries and walls
                if (0 <= next_state[0] < self.env.size and 
                    0 <= next_state[1] < self.env.size and
                    self.env.maze[next_state] == 0):
                    
                    # Calculate reward
                    reward = -0.01  # Step penalty
                    if next_state == self.env.goal:
                        reward = 100
                    elif next_state in self.env.treasure_positions:
                        reward = 10
                    elif next_state in self.env.trap_positions:
                        reward = -10
                    
                    # Update agent
                    self.agent.update_q_value(state, action, reward, next_state)
                    
                    # Move to next state
                    self.env.current_pos = next_state
                    self.env.visited_states.add(next_state)
                    self.agent.exploration_memory[next_state] += 1
                    episode_path.append(next_state)
                    
                    state = next_state
                    
                    # Check if goal reached
                    if state == self.env.goal:
                        break
                else:
                    # Invalid move - stay in place with penalty
                    reward = -1
                    self.agent.update_q_value(state, action, reward, state)
            
            # Collect episode insights
            episode_insights = [insight for insight in self.agent.insights 
                              if insight.episode == episode]
            
            self.episode_data.append({
                'episode': episode,
                'path': episode_path,
                'insights': episode_insights,
                'goal_reached': state == self.env.goal
            })
            
            # Update epsilon
            self.agent.epsilon = max(self.agent.epsilon_min, 
                                   self.agent.epsilon * self.agent.epsilon_decay)
        
        # Create visualizations
        self._plot_maze_with_path()
        self._plot_insight_timeline()
        self._plot_performance_metrics()
        self._plot_exploration_heatmap()
        
        plt.tight_layout()
        plt.show()
    
    def _plot_maze_with_path(self):
        """迷路とパスの可視化"""
        ax = self.axes[0, 0]
        ax.set_title('🗺️ 迷路探索パス', fontsize=14, fontweight='bold')
        
        # Draw maze
        maze_display = self.env.maze.copy()
        im = ax.imshow(maze_display, cmap='binary', alpha=0.8)
        
        # Mark special positions
        ax.scatter(self.env.start[1], self.env.start[0], c='green', s=200, marker='o', label='Start')
        ax.scatter(self.env.goal[1], self.env.goal[0], c='red', s=200, marker='*', label='Goal')
        
        # Mark treasures
        for treasure in self.env.treasure_positions:
            ax.scatter(treasure[1], treasure[0], c='gold', s=150, marker='D', label='Treasure')
        
        # Mark traps
        for trap in self.env.trap_positions:
            ax.scatter(trap[1], trap[0], c='purple', s=150, marker='X', label='Trap')
        
        # Draw successful paths
        successful_episodes = [ep for ep in self.episode_data if ep['goal_reached']]
        if successful_episodes:
            best_episode = min(successful_episodes, key=lambda x: len(x['path']))
            path = best_episode['path']
            
            # Draw path
            for i in range(len(path)-1):
                y1, x1 = path[i]
                y2, x2 = path[i+1]
                ax.arrow(x1, y1, x2-x1, y2-y1, 
                        head_width=0.3, head_length=0.3, 
                        fc='blue', ec='blue', alpha=0.7)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def _plot_insight_timeline(self):
        """洞察生成タイムライン"""
        ax = self.axes[0, 1]
        ax.set_title('💡 洞察生成タイムライン', fontsize=14, fontweight='bold')
        
        episodes = []
        insight_counts = []
        insight_types = {'strategic_breakthrough': [], 'goal_discovery': [], 'exploration_insight': []}
        
        for ep_data in self.episode_data:
            episodes.append(ep_data['episode'])
            insights = ep_data['insights']
            insight_counts.append(len(insights))
            
            # Count by type
            type_counts = {'strategic_breakthrough': 0, 'goal_discovery': 0, 'exploration_insight': 0}
            for insight in insights:
                type_counts[insight.insight_type] += 1
            
            for itype in insight_types:
                insight_types[itype].append(type_counts[itype])
        
        # Stacked bar chart
        bottom = np.zeros(len(episodes))
        colors = {'strategic_breakthrough': '#FF6B6B', 'goal_discovery': '#4ECDC4', 'exploration_insight': '#45B7D1'}
        
        for itype, counts in insight_types.items():
            ax.bar(episodes, counts, bottom=bottom, label=itype.replace('_', ' ').title(), 
                  color=colors[itype], alpha=0.8)
            bottom += counts
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Number of Insights')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_metrics(self):
        """パフォーマンス指標"""
        ax = self.axes[1, 0]
        ax.set_title('📊 学習パフォーマンス', fontsize=14, fontweight='bold')
        
        episodes = [ep['episode'] for ep in self.episode_data]
        success_rate = []
        path_efficiency = []
        
        window_size = 10
        for i in range(len(episodes)):
            start_idx = max(0, i - window_size + 1)
            window_data = self.episode_data[start_idx:i+1]
            
            # Success rate
            successes = sum(1 for ep in window_data if ep['goal_reached'])
            success_rate.append(successes / len(window_data))
            
            # Path efficiency (shorter paths are better)
            successful_paths = [len(ep['path']) for ep in window_data if ep['goal_reached']]
            if successful_paths:
                path_efficiency.append(1.0 / np.mean(successful_paths))
            else:
                path_efficiency.append(0)
        
        ax.plot(episodes, success_rate, label='Success Rate', color='green', linewidth=2)
        ax.plot(episodes, path_efficiency, label='Path Efficiency', color='blue', linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_exploration_heatmap(self):
        """探索ヒートマップ"""
        ax = self.axes[1, 1]
        ax.set_title('🔥 探索ヒートマップ', fontsize=14, fontweight='bold')
        
        # Create visit count matrix
        visit_counts = np.zeros((self.env.size, self.env.size))
        
        for ep_data in self.episode_data:
            for pos in ep_data['path']:
                visit_counts[pos[0], pos[1]] += 1
        
        # Mask walls
        visit_counts = np.ma.masked_where(self.env.maze == 1, visit_counts)
        
        im = ax.imshow(visit_counts, cmap='YlOrRd', alpha=0.8)
        
        # Mark special positions
        ax.scatter(self.env.start[1], self.env.start[0], c='green', s=100, marker='o', edgecolors='black')
        ax.scatter(self.env.goal[1], self.env.goal[0], c='red', s=100, marker='*', edgecolors='black')
        
        plt.colorbar(im, ax=ax, label='Visit Count')
        ax.set_xticks([])
        ax.set_yticks([])

def main():
    """メイン実行関数"""
    print("🧠 InsightSpike-AI 迷路探索可視化システム起動中...")
    print("=" * 60)
    
    # Initialize environment and agent
    env = DynamicMazeEnvironment(size=12, wall_density=0.25)
    agent = InsightSpikeAgent(env)
    
    # Create visualization system
    viz_system = MazeVisualizationSystem(env, agent)
    
    print("🗺️ 迷路環境生成完了")
    print(f"📍 Start: {env.start}, Goal: {env.goal}")
    print(f"💎 Treasures: {len(env.treasure_positions)}, ⚠️ Traps: {len(env.trap_positions)}")
    print()
    
    print("🚀 探索開始...")
    start_time = time.time()
    
    # Run visualization
    viz_system.create_static_visualization()
    
    end_time = time.time()
    
    # Print results
    print("\n" + "=" * 60)
    print("🎯 実験結果サマリー")
    print("=" * 60)
    print(f"⏱️ 実行時間: {end_time - start_time:.2f}秒")
    print(f"🧠 総洞察数: {len(agent.insights)}")
    print(f"🎯 成功率: {sum(1 for ep in viz_system.episode_data if ep['goal_reached']) / len(viz_system.episode_data) * 100:.1f}%")
    
    # Insight analysis
    insight_types = defaultdict(int)
    for insight in agent.insights:
        insight_types[insight.insight_type] += 1
    
    print("\n💡 洞察タイプ別分析:")
    for itype, count in insight_types.items():
        print(f"   {itype.replace('_', ' ').title()}: {count}")
    
    # Save results
    results_dir = "/Users/miyauchikazuyoshi/Documents/GitHub/InsightSpike-AI/experiments/results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{results_dir}/maze_exploration_visualization_{timestamp}.png", 
                dpi=300, bbox_inches='tight')
    
    print(f"\n💾 結果を保存しました: {results_dir}/maze_exploration_visualization_{timestamp}.png")

if __name__ == "__main__":
    main()
