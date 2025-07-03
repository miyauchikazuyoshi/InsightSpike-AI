#!/usr/bin/env python3
"""
Foundational Experiment: Intrinsic Motivation (ΔGED × ΔIG) - Improved Version
===========================================================================

This improved version addresses the following issues from the review:
1. Random seed control for reproducibility
2. Grid-World connectivity guarantee (solvable mazes)
3. Reward scale adjustment for fair baseline comparison
4. Intrinsic reward weight parameter validation
5. Statistical significance testing
6. Convergence speed and stability metrics

Major improvements:
- Fixed random seeds throughout
- Path connectivity check for Grid-World
- Adjustable reward scales
- Grid search for optimal intrinsic weight
- Paired t-tests for significance
- Enhanced metrics (convergence speed, stability)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import deque, defaultdict
import warnings
from scipy import stats
from itertools import product
import networkx as nx

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Environment imports
try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        GYM_AVAILABLE = True
        print("Using legacy gym. Consider upgrading to gymnasium.")
    except ImportError:
        GYM_AVAILABLE = False
        print("Neither gymnasium nor gym available. MountainCar will be skipped.")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Set PyTorch seeds
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

# InsightSpike-AI imports
try:
    from insightspike.algorithms.graph_edit_distance import GraphEditDistance
    from insightspike.algorithms.information_gain import InformationGain
    INSIGHTSPIKE_AVAILABLE = True
except ImportError:
    INSIGHTSPIKE_AVAILABLE = False
    print("InsightSpike-AI not available, using simplified implementations")


class ImprovedGridWorld:
    """Improved Grid-World with guaranteed solvability"""
    
    def __init__(self, size=8, num_obstacles=None, goal_reward=10.0):
        self.size = size
        self.num_obstacles = num_obstacles or size // 2
        self.goal_reward = goal_reward  # Adjustable goal reward
        self.step_penalty = -0.01
        self.collision_penalty = -0.1
        self.timeout_penalty = -1.0
        self.reset()
    
    def _check_path_exists(self, start, goal, obstacles):
        """Check if path exists from start to goal using BFS"""
        if start in obstacles or goal in obstacles:
            return False
        
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            current = queue.popleft()
            if current == goal:
                return True
            
            row, col = current
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_row, next_col = row + dr, col + dc
                next_pos = (next_row, next_col)
                
                if (0 <= next_row < self.size and 
                    0 <= next_col < self.size and
                    next_pos not in visited and 
                    next_pos not in obstacles):
                    
                    visited.add(next_pos)
                    queue.append(next_pos)
        
        return False
    
    def reset(self):
        """Reset environment with guaranteed solvable maze"""
        self.grid = np.zeros((self.size, self.size))
        self.start_pos = (0, 0)
        self.goal_pos = (self.size-1, self.size-1)
        
        # Generate obstacles ensuring path exists
        max_attempts = 100
        for _ in range(max_attempts):
            obstacle_positions = set()
            
            # Generate random obstacles
            while len(obstacle_positions) < self.num_obstacles:
                row = np.random.randint(self.size)
                col = np.random.randint(self.size)
                pos = (row, col)
                
                # Don't place on start or goal
                if pos != self.start_pos and pos != self.goal_pos:
                    obstacle_positions.add(pos)
            
            # Check if path exists
            if self._check_path_exists(self.start_pos, self.goal_pos, obstacle_positions):
                # Valid maze found
                for row, col in obstacle_positions:
                    self.grid[row, col] = -1
                break
        else:
            # If no valid maze found, create simple one with fewer obstacles
            print("Warning: Could not generate valid maze, using fewer obstacles")
            self.num_obstacles = max(1, self.num_obstacles // 2)
            return self.reset()
        
        self.current_pos = self.start_pos
        self.grid[self.goal_pos] = 1
        self.step_count = 0
        self.max_steps = self.size * self.size * 2
        self.visited_positions = set([self.current_pos])
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation"""
        state = np.zeros((self.size, self.size, 3))
        
        # Current position channel
        state[self.current_pos[0], self.current_pos[1], 0] = 1
        
        # Obstacles channel
        state[:, :, 1] = (self.grid == -1).astype(float)
        
        # Goal channel
        state[self.goal_pos[0], self.goal_pos[1], 2] = 1
        
        return state.flatten()
    
    def step(self, action):
        """Execute action and return (next_state, reward, done, info)"""
        # Actions: 0=up, 1=right, 2=down, 3=left
        action_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        
        if action not in action_map:
            raise ValueError(f"Invalid action: {action}")
        
        dr, dc = action_map[action]
        new_row = self.current_pos[0] + dr
        new_col = self.current_pos[1] + dc
        
        # Check boundaries and obstacles
        if (0 <= new_row < self.size and 
            0 <= new_col < self.size and 
            self.grid[new_row, new_col] != -1):
            
            self.current_pos = (new_row, new_col)
            self.visited_positions.add(self.current_pos)
            reward = self.step_penalty
        else:
            # Hit wall or obstacle
            reward = self.collision_penalty
        
        self.step_count += 1
        
        # Check terminal conditions
        done = False
        info = {}
        
        if self.current_pos == self.goal_pos:
            reward = self.goal_reward
            done = True
            info['success'] = True
        elif self.step_count >= self.max_steps:
            reward = self.timeout_penalty
            done = True
            info['success'] = False
        else:
            info['success'] = False
        
        info['steps'] = self.step_count
        info['coverage'] = len(self.visited_positions) / (self.size * self.size)
        
        return self._get_state(), reward, done, info
    
    @property
    def state_space_size(self):
        return self.size * self.size * 3
    
    @property
    def action_space_size(self):
        return 4


class OptimizedIntrinsicMotivationAgent:
    """Agent with optimizable intrinsic motivation parameters"""
    
    def __init__(self, state_size, action_size, 
                 use_ged=True, use_ig=True, 
                 intrinsic_weight=0.1,
                 learning_rate=0.001):
        
        self.state_size = state_size
        self.action_size = action_size
        self.use_ged = use_ged
        self.use_ig = use_ig
        self.intrinsic_weight = intrinsic_weight
        
        # Q-Network
        self.q_network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # State tracking for intrinsic motivation
        self.state_visits = defaultdict(int)
        self.state_history = deque(maxlen=100)
        
        # Initialize InsightSpike components if available
        if INSIGHTSPIKE_AVAILABLE:
            self.ged_calculator = GraphEditDistance()
            self.ig_calculator = InformationGain()
    
    def _state_to_graph(self, state):
        """Convert state to graph representation"""
        # Simple graph representation for Grid-World state
        G = nx.Graph()
        
        # Reshape state back to grid
        state_3d = state.reshape(-1, 3)
        grid_size = int(np.sqrt(len(state_3d)))
        
        # Find agent position
        agent_pos = np.unravel_index(np.argmax(state_3d[:, 0]), (grid_size, grid_size))
        G.add_node('agent', pos=agent_pos)
        
        # Add obstacle nodes
        obstacle_indices = np.where(state_3d[:, 1] > 0)[0]
        for idx in obstacle_indices:
            pos = np.unravel_index(idx, (grid_size, grid_size))
            G.add_node(f'obstacle_{idx}', pos=pos)
            G.add_edge('agent', f'obstacle_{idx}', weight=np.linalg.norm(np.array(agent_pos) - np.array(pos)))
        
        # Add goal node
        goal_pos = np.unravel_index(np.argmax(state_3d[:, 2]), (grid_size, grid_size))
        G.add_node('goal', pos=goal_pos)
        G.add_edge('agent', 'goal', weight=np.linalg.norm(np.array(agent_pos) - np.array(goal_pos)))
        
        return G
    
    def _calculate_ged(self, state1, state2):
        """Calculate Graph Edit Distance between states"""
        if not self.use_ged:
            return 1.0
        
        if INSIGHTSPIKE_AVAILABLE and False:  # Disable for now due to API issues
            # Use actual InsightSpike GED
            g1 = self._state_to_graph(state1)
            g2 = self._state_to_graph(state2)
            result = self.ged_calculator.calculate(g1, g2)
            return result.ged_value
        else:
            # Simplified GED based on state difference
            diff = np.linalg.norm(state1 - state2)
            return diff / (np.linalg.norm(state1) + np.linalg.norm(state2) + 1e-8)
    
    def _calculate_ig(self, state):
        """Calculate Information Gain for state"""
        if not self.use_ig:
            return 1.0
        
        # Convert state to hashable key
        state_key = tuple(state.flatten())
        old_count = self.state_visits[state_key]
        self.state_visits[state_key] += 1
        
        if INSIGHTSPIKE_AVAILABLE and False:  # Disable for now due to API issues
            # Use actual InsightSpike IG - needs before/after data
            visit_counts_before = np.array(list(self.state_visits.values()))
            visit_counts_before[list(self.state_visits.keys()).index(state_key)] = old_count
            visit_counts_after = np.array(list(self.state_visits.values()))
            result = self.ig_calculator.calculate(visit_counts_before, visit_counts_after)
            return result.ig_value
        else:
            # Simplified IG based on visit frequency
            visit_count = self.state_visits[state_key]
            return 1.0 / (1.0 + visit_count)
    
    def _calculate_intrinsic_reward(self, current_state, previous_state=None):
        """Calculate intrinsic reward (ΔGED × ΔIG)"""
        if not self.use_ged and not self.use_ig:
            return 0.0
        
        if previous_state is None:
            if len(self.state_history) == 0:
                return 0.0
            previous_state = self.state_history[-1]
        
        delta_ged = self._calculate_ged(current_state, previous_state)
        delta_ig = self._calculate_ig(current_state)
        
        intrinsic_reward = delta_ged * delta_ig
        return intrinsic_reward
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done, info=None):
        """Store experience with intrinsic reward"""
        intrinsic_reward = self._calculate_intrinsic_reward(next_state, state)
        total_reward = reward + self.intrinsic_weight * intrinsic_reward
        
        self.memory.append((state, action, total_reward, next_state, done))
        self.state_history.append(next_state)
    
    def replay(self, batch_size=32):
        """Train the network on experiences"""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.q_network(next_states).max(1)[0]
            target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def optimize_intrinsic_weight(env_class, env_params, episodes_per_trial=100, n_trials=5):
    """Find optimal intrinsic reward weight via grid search"""
    
    weight_candidates = [0.0, 0.05, 0.1, 0.2, 0.5]
    best_weight = 0.1
    best_score = -float('inf')
    
    print("Optimizing intrinsic reward weight...")
    
    for weight in weight_candidates:
        print(f"\nTesting weight: {weight}")
        
        scores = []
        for trial in range(n_trials):
            # Set seed for each trial
            seed = RANDOM_SEED + trial
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            env = env_class(**env_params)
            agent = OptimizedIntrinsicMotivationAgent(
                state_size=env.state_space_size,
                action_size=env.action_space_size,
                intrinsic_weight=weight
            )
            
            # Run episodes
            successes = 0
            for episode in range(episodes_per_trial):
                state = env.reset()
                done = False
                
                while not done:
                    action = agent.act(state)
                    next_state, reward, done, info = env.step(action)
                    agent.remember(state, action, reward, next_state, done, info)
                    
                    if len(agent.memory) > 32:
                        agent.replay()
                    
                    state = next_state
                    
                    if done and info.get('success', False):
                        successes += 1
            
            score = successes / episodes_per_trial
            scores.append(score)
        
        avg_score = np.mean(scores)
        print(f"  Average success rate: {avg_score:.3f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_weight = weight
    
    print(f"\nOptimal intrinsic weight: {best_weight}")
    return best_weight


def run_improved_grid_world_experiment(episodes=300, trials=5, optimize_weight=True):
    """Run improved Grid-World experiment with all fixes"""
    
    env_params = {'size': 8, 'goal_reward': 10.0}
    
    # Optimize intrinsic weight if requested
    if optimize_weight:
        optimal_weight = optimize_intrinsic_weight(
            ImprovedGridWorld, env_params, 
            episodes_per_trial=50, n_trials=3
        )
    else:
        optimal_weight = 0.1
    
    configurations = [
        {"name": "Full (ΔGED × ΔIG)", "use_ged": True, "use_ig": True, "weight": optimal_weight},
        {"name": "GED Only", "use_ged": True, "use_ig": False, "weight": optimal_weight},
        {"name": "IG Only", "use_ged": False, "use_ig": True, "weight": optimal_weight},
        {"name": "Baseline", "use_ged": False, "use_ig": False, "weight": 0.0}
    ]
    
    results = {}
    
    for config in configurations:
        print(f"\nRunning configuration: {config['name']}")
        
        config_results = {
            "success_rates": [],
            "convergence_episodes": [],  # Episode where 50% success rate achieved
            "final_performance": [],     # Average of last 20% episodes
            "stability": [],            # Std dev of last 20% episodes
            "learning_curves": [],
            "episode_lengths": [],
            "coverage_rates": []        # How much of the grid was explored
        }
        
        for trial in range(trials):
            print(f"  Trial {trial + 1}/{trials}")
            
            # Set seed for reproducibility
            seed = RANDOM_SEED + trial * 100
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            
            env = ImprovedGridWorld(**env_params)
            agent = OptimizedIntrinsicMotivationAgent(
                state_size=env.state_space_size,
                action_size=env.action_space_size,
                use_ged=config["use_ged"],
                use_ig=config["use_ig"],
                intrinsic_weight=config["weight"]
            )
            
            trial_rewards = []
            trial_successes = []
            trial_lengths = []
            trial_coverage = []
            convergence_episode = episodes  # Default to max if never converges
            
            for episode in range(episodes):
                state = env.reset()
                total_reward = 0
                done = False
                
                while not done:
                    action = agent.act(state)
                    next_state, reward, done, info = env.step(action)
                    agent.remember(state, action, reward, next_state, done, info)
                    
                    state = next_state
                    total_reward += reward
                    
                    if done:
                        trial_successes.append(1 if info['success'] else 0)
                        trial_lengths.append(info['steps'])
                        trial_coverage.append(info['coverage'])
                        break
                
                trial_rewards.append(total_reward)
                
                # Train agent
                if len(agent.memory) > 32:
                    for _ in range(4):  # Multiple replay steps
                        agent.replay()
                
                # Check convergence (moving average > 0.5)
                if episode >= 10:
                    recent_success_rate = np.mean(trial_successes[-10:])
                    if recent_success_rate >= 0.5 and convergence_episode == episodes:
                        convergence_episode = episode
            
            # Calculate metrics
            config_results["success_rates"].append(np.mean(trial_successes))
            config_results["convergence_episodes"].append(convergence_episode)
            
            # Final performance (last 20% of episodes)
            final_start = int(0.8 * episodes)
            final_rewards = trial_rewards[final_start:]
            config_results["final_performance"].append(np.mean(final_rewards))
            config_results["stability"].append(np.std(final_rewards))
            
            config_results["learning_curves"].append(trial_rewards)
            config_results["episode_lengths"].append(trial_lengths)
            config_results["coverage_rates"].append(np.mean(trial_coverage))
        
        results[config["name"]] = config_results
    
    return results, optimal_weight


def run_improved_mountain_car_experiment(episodes=500, trials=3, optimal_weight=0.1):
    """Run improved MountainCar experiment"""
    
    if not GYM_AVAILABLE:
        print("Gym not available, skipping MountainCar")
        return {}
    
    configurations = [
        {"name": "Full (ΔGED × ΔIG)", "use_ged": True, "use_ig": True, "weight": optimal_weight},
        {"name": "GED Only", "use_ged": True, "use_ig": False, "weight": optimal_weight},
        {"name": "IG Only", "use_ged": False, "use_ig": True, "weight": optimal_weight},
        {"name": "Baseline", "use_ged": False, "use_ig": False, "weight": 0.0}
    ]
    
    results = {}
    
    for config in configurations:
        print(f"\nRunning MountainCar - {config['name']}")
        
        config_results = {
            "success_rates": [],
            "average_episode_lengths": [],
            "convergence_episodes": [],
            "learning_curves": []
        }
        
        for trial in range(trials):
            print(f"  Trial {trial + 1}/{trials}")
            
            # Set seeds
            seed = RANDOM_SEED + trial * 200
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            env = gym.make('MountainCar-v0')
            if hasattr(env, 'seed'):
                env.seed(seed)
            
            agent = OptimizedIntrinsicMotivationAgent(
                state_size=2,  # MountainCar has 2D state
                action_size=3,  # 3 actions
                use_ged=config["use_ged"],
                use_ig=config["use_ig"],
                intrinsic_weight=config["weight"]
            )
            
            successes = 0
            episode_lengths = []
            convergence_episode = episodes
            
            for episode in range(episodes):
                state = env.reset()
                if isinstance(state, tuple):
                    state = state[0]  # Handle new gym API
                
                total_reward = 0
                steps = 0
                
                for step in range(200):  # MountainCar limit
                    action = agent.act(state)
                    result = env.step(action)
                    
                    # Handle different gym API versions
                    if len(result) == 4:
                        next_state, reward, done, info = result
                    else:
                        next_state, reward, done, truncated, info = result
                        done = done or truncated
                    
                    agent.remember(state, action, reward, next_state, done, info)
                    
                    state = next_state
                    total_reward += reward
                    steps += 1
                    
                    if done:
                        if steps < 200:  # Success if finished before timeout
                            successes += 1
                        break
                
                episode_lengths.append(steps)
                
                # Train
                if len(agent.memory) > 32:
                    for _ in range(4):
                        agent.replay()
                
                # Check convergence
                if episode >= 10:
                    recent_success_rate = successes / (episode + 1)
                    if recent_success_rate >= 0.5 and convergence_episode == episodes:
                        convergence_episode = episode
            
            config_results["success_rates"].append(successes / episodes)
            config_results["average_episode_lengths"].append(np.mean(episode_lengths))
            config_results["convergence_episodes"].append(convergence_episode)
            config_results["learning_curves"].append(episode_lengths)
        
        results[config["name"]] = config_results
    
    return results


def perform_statistical_analysis(results):
    """Perform statistical tests on experimental results"""
    
    stats_results = {}
    
    # Extract success rates for each configuration
    config_names = list(results.keys())
    success_rates = {
        config: results[config]["success_rates"] 
        for config in config_names
    }
    
    # Pairwise t-tests
    print("\nStatistical Analysis - Paired t-tests:")
    print("=" * 50)
    
    for i, config1 in enumerate(config_names):
        for config2 in config_names[i+1:]:
            data1 = success_rates[config1]
            data2 = success_rates[config2]
            
            if len(data1) == len(data2) and len(data1) > 1:
                t_stat, p_value = stats.ttest_rel(data1, data2)
                
                # Cohen's d
                diff = np.array(data1) - np.array(data2)
                cohen_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)
                
                print(f"\n{config1} vs {config2}:")
                print(f"  t-statistic: {t_stat:.3f}")
                print(f"  p-value: {p_value:.4f}")
                print(f"  Cohen's d: {cohen_d:.3f}")
                print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
                
                stats_results[f"{config1}_vs_{config2}"] = {
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "cohen_d": cohen_d,
                    "significant": p_value < 0.05
                }
    
    # ANOVA across all groups
    all_data = [success_rates[config] for config in config_names]
    f_stat, p_value = stats.f_oneway(*all_data)
    
    print(f"\nOne-way ANOVA:")
    print(f"  F-statistic: {f_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    stats_results["anova"] = {
        "f_stat": f_stat,
        "p_value": p_value,
        "significant": p_value < 0.05
    }
    
    return stats_results


def create_improved_visualizations(grid_results, mountain_results, stats_results, output_dir):
    """Create comprehensive visualizations"""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Grid-World Results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Success rates comparison
    ax = axes[0, 0]
    configs = list(grid_results.keys())
    success_means = [np.mean(grid_results[c]["success_rates"]) for c in configs]
    success_stds = [np.std(grid_results[c]["success_rates"]) for c in configs]
    
    bars = ax.bar(configs, success_means, yerr=success_stds, capsize=5)
    ax.set_ylabel("Success Rate")
    ax.set_title("Grid-World: Success Rate Comparison")
    ax.set_ylim(0, 1.1)
    
    # Add significance markers
    y_max = max(success_means) + max(success_stds) + 0.1
    for i, c1 in enumerate(configs):
        for j, c2 in enumerate(configs[i+1:], i+1):
            key = f"{c1}_vs_{c2}"
            if key in stats_results and stats_results[key]["significant"]:
                ax.plot([i, j], [y_max, y_max], 'k-')
                ax.text((i+j)/2, y_max + 0.02, '*', ha='center', fontsize=14)
    
    # Convergence speed
    ax = axes[0, 1]
    conv_means = [np.mean(grid_results[c]["convergence_episodes"]) for c in configs]
    conv_stds = [np.std(grid_results[c]["convergence_episodes"]) for c in configs]
    
    bars = ax.bar(configs, conv_means, yerr=conv_stds, capsize=5)
    ax.set_ylabel("Episodes to Convergence")
    ax.set_title("Grid-World: Convergence Speed")
    
    # Stability (lower is better)
    ax = axes[0, 2]
    stability_means = [np.mean(grid_results[c]["stability"]) for c in configs]
    
    bars = ax.bar(configs, stability_means)
    ax.set_ylabel("Performance Std Dev")
    ax.set_title("Grid-World: Learning Stability")
    
    # Learning curves
    ax = axes[1, 0]
    for config in configs:
        curves = grid_results[config]["learning_curves"]
        mean_curve = np.mean(curves, axis=0)
        
        # Smooth curve with moving average
        window = 10
        smoothed = np.convolve(mean_curve, np.ones(window)/window, mode='valid')
        episodes = np.arange(len(smoothed))
        
        ax.plot(episodes, smoothed, label=config, linewidth=2)
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Grid-World: Learning Curves")
    ax.legend()
    
    # Coverage analysis
    ax = axes[1, 1]
    coverage_means = [np.mean(grid_results[c]["coverage_rates"]) for c in configs]
    
    bars = ax.bar(configs, coverage_means)
    ax.set_ylabel("Average Grid Coverage")
    ax.set_title("Grid-World: Exploration Coverage")
    ax.set_ylim(0, 1)
    
    # MountainCar results (if available)
    ax = axes[1, 2]
    if mountain_results:
        mc_success_means = [np.mean(mountain_results[c]["success_rates"]) for c in configs]
        mc_success_stds = [np.std(mountain_results[c]["success_rates"]) for c in configs]
        
        bars = ax.bar(configs, mc_success_means, yerr=mc_success_stds, capsize=5)
        ax.set_ylabel("Success Rate")
        ax.set_title("MountainCar: Success Rate")
    else:
        ax.text(0.5, 0.5, "MountainCar not available", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title("MountainCar: N/A")
    
    plt.tight_layout()
    plt.savefig(output_dir / "intrinsic_motivation_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed parameter analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Effect of intrinsic weight
    ax = axes[0]
    ax.text(0.5, 0.5, "Intrinsic weight optimization results\nwould be shown here", 
            ha='center', va='center', transform=ax.transAxes)
    ax.set_title("Effect of Intrinsic Weight Parameter")
    
    # Statistical significance heatmap
    ax = axes[1]
    n_configs = len(configs)
    p_matrix = np.ones((n_configs, n_configs))
    
    for i, c1 in enumerate(configs):
        for j, c2 in enumerate(configs):
            if i != j:
                key = f"{c1}_vs_{c2}" if i < j else f"{c2}_vs_{c1}"
                if key in stats_results:
                    p_matrix[i, j] = stats_results[key]["p_value"]
    
    im = ax.imshow(p_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.1)
    ax.set_xticks(range(n_configs))
    ax.set_yticks(range(n_configs))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_yticklabels(configs)
    ax.set_title("Statistical Significance (p-values)")
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved!")


def main():
    """Run the complete improved experiment"""
    
    output_dir = Path("experiments/foundational_intrinsic_motivation/results_improved")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Intrinsic Motivation Experiment - Improved Version")
    print("=" * 60)
    
    # Run Grid-World experiment
    print("\n1. Running Grid-World Experiment...")
    grid_results, optimal_weight = run_improved_grid_world_experiment(
        episodes=300, trials=5, optimize_weight=True
    )
    
    # Run MountainCar experiment
    print("\n2. Running MountainCar Experiment...")
    mountain_results = run_improved_mountain_car_experiment(
        episodes=500, trials=3, optimal_weight=optimal_weight
    )
    
    # Statistical analysis
    print("\n3. Performing Statistical Analysis...")
    all_results = grid_results
    stats_results = perform_statistical_analysis(all_results)
    
    # Save results
    results_data = {
        "grid_world": {
            config: {
                "success_rate_mean": float(np.mean(data["success_rates"])),
                "success_rate_std": float(np.std(data["success_rates"])),
                "convergence_mean": float(np.mean(data["convergence_episodes"])),
                "stability_mean": float(np.mean(data["stability"])),
                "coverage_mean": float(np.mean(data["coverage_rates"]))
            }
            for config, data in grid_results.items()
        },
        "mountain_car": {
            config: {
                "success_rate_mean": float(np.mean(data["success_rates"])),
                "success_rate_std": float(np.std(data["success_rates"])),
                "avg_episode_length": float(np.mean([np.mean(lengths) for lengths in data["learning_curves"]]))
            }
            for config, data in mountain_results.items()
        } if mountain_results else {},
        "statistical_tests": {
            k: {
                "t_stat": float(v["t_stat"]),
                "p_value": float(v["p_value"]),
                "cohen_d": float(v["cohen_d"]),
                "significant": bool(v["significant"])
            }
            for k, v in stats_results.items() if isinstance(v, dict) and "t_stat" in v
        },
        "optimal_intrinsic_weight": float(optimal_weight),
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "random_seed": RANDOM_SEED,
            "grid_world_episodes": 300,
            "mountain_car_episodes": 500
        }
    }
    
    with open(output_dir / "experiment_results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Create visualizations
    print("\n4. Creating Visualizations...")
    create_improved_visualizations(grid_results, mountain_results, stats_results, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    print("\nGrid-World Results:")
    for config in grid_results:
        mean_success = np.mean(grid_results[config]["success_rates"])
        std_success = np.std(grid_results[config]["success_rates"])
        print(f"  {config}: {mean_success:.3f} ± {std_success:.3f}")
    
    print(f"\nOptimal Intrinsic Weight: {optimal_weight}")
    
    print("\nKey Findings:")
    # Find best performing configuration
    best_config = max(grid_results.keys(), 
                     key=lambda c: np.mean(grid_results[c]["success_rates"]))
    print(f"  Best performing: {best_config}")
    
    # Check if Full is significantly better than Baseline
    if "Full (ΔGED × ΔIG)_vs_Baseline" in stats_results:
        if stats_results["Full (ΔGED × ΔIG)_vs_Baseline"]["significant"]:
            print("  Full intrinsic motivation significantly outperforms baseline (p < 0.05)")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()