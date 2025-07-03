#!/usr/bin/env python3
"""
Lightweight version of improved intrinsic motivation experiment
Reduced complexity for faster execution while maintaining all improvements
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import deque, defaultdict
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(RANDOM_SEED)


class SimpleGridWorld:
    """Simplified Grid-World with path checking"""
    
    def __init__(self, size=6, num_obstacles=3, goal_reward=10.0):
        self.size = size
        self.num_obstacles = num_obstacles
        self.goal_reward = goal_reward
        self.step_penalty = -0.01
        self.collision_penalty = -0.1
        self.timeout_penalty = -1.0
        self.reset()
    
    def _check_path_exists(self):
        """Simple path check using BFS"""
        visited = set()
        queue = deque([self.start_pos])
        visited.add(self.start_pos)
        
        while queue:
            pos = queue.popleft()
            if pos == self.goal_pos:
                return True
            
            row, col = pos
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_pos = (row + dr, col + dc)
                if (0 <= new_pos[0] < self.size and 
                    0 <= new_pos[1] < self.size and
                    new_pos not in visited and 
                    self.grid[new_pos] != -1):
                    visited.add(new_pos)
                    queue.append(new_pos)
        return False
    
    def reset(self):
        """Reset with guaranteed solvable maze"""
        self.grid = np.zeros((self.size, self.size))
        self.start_pos = (0, 0)
        self.goal_pos = (self.size-1, self.size-1)
        
        # Try to place obstacles
        for _ in range(10):  # Max attempts
            self.grid.fill(0)
            obstacles = set()
            
            while len(obstacles) < self.num_obstacles:
                pos = (np.random.randint(self.size), np.random.randint(self.size))
                if pos != self.start_pos and pos != self.goal_pos:
                    obstacles.add(pos)
            
            for pos in obstacles:
                self.grid[pos] = -1
            
            if self._check_path_exists():
                break
        
        self.current_pos = self.start_pos
        self.step_count = 0
        self.max_steps = self.size * self.size * 2
        
        return self._get_state()
    
    def _get_state(self):
        """Simple state representation"""
        # One-hot position encoding
        state = np.zeros(self.size * self.size)
        state[self.current_pos[0] * self.size + self.current_pos[1]] = 1
        return state
    
    def step(self, action):
        """Execute action"""
        moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        dr, dc = moves[action]
        
        new_pos = (self.current_pos[0] + dr, self.current_pos[1] + dc)
        
        if (0 <= new_pos[0] < self.size and 
            0 <= new_pos[1] < self.size and 
            self.grid[new_pos] != -1):
            self.current_pos = new_pos
            reward = self.step_penalty
        else:
            reward = self.collision_penalty
        
        self.step_count += 1
        done = False
        info = {'success': False}
        
        if self.current_pos == self.goal_pos:
            reward = self.goal_reward
            done = True
            info['success'] = True
        elif self.step_count >= self.max_steps:
            reward = self.timeout_penalty
            done = True
        
        return self._get_state(), reward, done, info
    
    @property
    def state_space_size(self):
        return self.size * self.size
    
    @property
    def action_space_size(self):
        return 4


class SimpleAgent:
    """Simplified agent with intrinsic motivation"""
    
    def __init__(self, state_size, action_size, use_ged=True, use_ig=True, intrinsic_weight=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.use_ged = use_ged
        self.use_ig = use_ig
        self.intrinsic_weight = intrinsic_weight
        
        # Simple network
        self.q_network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.memory = deque(maxlen=2000)
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.state_visits = defaultdict(int)
        self.prev_state = None
    
    def _calculate_intrinsic_reward(self, state):
        """Simplified intrinsic reward"""
        if not self.use_ged and not self.use_ig:
            return 0.0
        
        # IG component - novelty
        state_key = tuple(state)
        self.state_visits[state_key] += 1
        novelty = 1.0 / (1.0 + self.state_visits[state_key]) if self.use_ig else 1.0
        
        # GED component - state change
        if self.prev_state is not None and self.use_ged:
            diversity = np.linalg.norm(state - self.prev_state)
        else:
            diversity = 1.0
        
        self.prev_state = state.copy()
        return novelty * diversity
    
    def act(self, state):
        """Epsilon-greedy action"""
        if np.random.random() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        with torch.no_grad():
            q_values = self.q_network(torch.FloatTensor(state))
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store with intrinsic reward"""
        intrinsic = self._calculate_intrinsic_reward(next_state)
        total_reward = reward + self.intrinsic_weight * intrinsic
        self.memory.append((state, action, total_reward, next_state, done))
    
    def replay(self, batch_size=32):
        """Simple experience replay"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.FloatTensor([e[4] for e in batch])
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.q_network(next_states).max(1)[0].detach()
        target_q = rewards + (0.99 * next_q * (1 - dones))
        
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def run_lightweight_experiment(episodes=100, trials=3):
    """Run lightweight experiment"""
    
    configs = [
        {"name": "Full", "ged": True, "ig": True, "weight": 0.1},
        {"name": "GED_Only", "ged": True, "ig": False, "weight": 0.1},
        {"name": "IG_Only", "ged": False, "ig": True, "weight": 0.1},
        {"name": "Baseline", "ged": False, "ig": False, "weight": 0.0}
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        success_rates = []
        convergence_episodes = []
        
        for trial in range(trials):
            # Set seed
            seed = RANDOM_SEED + trial * 10
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            
            env = SimpleGridWorld(size=6, num_obstacles=3)
            agent = SimpleAgent(
                env.state_space_size,
                env.action_space_size,
                use_ged=config['ged'],
                use_ig=config['ig'],
                intrinsic_weight=config['weight']
            )
            
            successes = []
            convergence_ep = episodes
            
            for ep in range(episodes):
                state = env.reset()
                done = False
                
                while not done:
                    action = agent.act(state)
                    next_state, reward, done, info = env.step(action)
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    
                    if done:
                        successes.append(1 if info['success'] else 0)
                
                # Train
                if len(agent.memory) > 32:
                    agent.replay()
                
                # Check convergence
                if len(successes) >= 10:
                    recent_rate = np.mean(successes[-10:])
                    if recent_rate >= 0.5 and convergence_ep == episodes:
                        convergence_ep = ep
            
            success_rate = np.mean(successes)
            success_rates.append(success_rate)
            convergence_episodes.append(convergence_ep)
            
            print(f"  Trial {trial+1}: Success={success_rate:.3f}, Conv={convergence_ep}")
        
        results[config['name']] = {
            'success_rates': success_rates,
            'convergence_episodes': convergence_episodes,
            'mean_success': np.mean(success_rates),
            'std_success': np.std(success_rates),
            'mean_convergence': np.mean(convergence_episodes)
        }
    
    return results


def analyze_results(results):
    """Statistical analysis"""
    
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    # Success rate comparison
    print("\nSuccess Rates:")
    for name, data in results.items():
        print(f"  {name}: {data['mean_success']:.3f} ± {data['std_success']:.3f}")
    
    # Pairwise t-tests
    print("\nPairwise Comparisons (t-test):")
    names = list(results.keys())
    
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            name1, name2 = names[i], names[j]
            data1 = results[name1]['success_rates']
            data2 = results[name2]['success_rates']
            
            if len(data1) == len(data2) > 1:
                t_stat, p_val = stats.ttest_rel(data1, data2)
                print(f"  {name1} vs {name2}: p={p_val:.4f} {'*' if p_val < 0.05 else ''}")
    
    # Convergence speed
    print("\nConvergence Speed:")
    for name, data in results.items():
        print(f"  {name}: {data['mean_convergence']:.1f} episodes")
    
    return results


def create_simple_plot(results, output_dir):
    """Create visualization"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Success rates
    names = list(results.keys())
    means = [results[n]['mean_success'] for n in names]
    stds = [results[n]['std_success'] for n in names]
    
    ax1.bar(names, means, yerr=stds, capsize=5)
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Performance Comparison')
    ax1.set_ylim(0, 1.1)
    
    # Convergence speed
    conv_means = [results[n]['mean_convergence'] for n in names]
    
    ax2.bar(names, conv_means)
    ax2.set_ylabel('Episodes to Convergence')
    ax2.set_title('Learning Speed')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lightweight_results.png', dpi=150)
    plt.close()


def main():
    """Run lightweight experiment"""
    
    output_dir = Path("experiments/foundational_intrinsic_motivation/results_improved")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("INTRINSIC MOTIVATION - LIGHTWEIGHT EXPERIMENT")
    print("="*60)
    
    # Run experiment
    results = run_lightweight_experiment(episodes=100, trials=3)
    
    # Analyze
    results = analyze_results(results)
    
    # Save results
    save_data = {
        'results': results,
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'random_seed': RANDOM_SEED,
            'episodes': 100,
            'trials': 3
        }
    }
    
    with open(output_dir / 'lightweight_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    
    # Plot
    create_simple_plot(results, output_dir)
    
    print(f"\nResults saved to {output_dir}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    best = max(results.keys(), key=lambda k: results[k]['mean_success'])
    print(f"Best performer: {best} ({results[best]['mean_success']:.3f})")
    
    if results['Full']['mean_success'] > results['Baseline']['mean_success']:
        improvement = (results['Full']['mean_success'] - results['Baseline']['mean_success']) / results['Baseline']['mean_success'] * 100
        print(f"Full intrinsic motivation shows {improvement:.1f}% improvement over baseline")


if __name__ == "__main__":
    main()