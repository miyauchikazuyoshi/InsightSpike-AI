#!/usr/bin/env python3
"""
Simplified complex maze experiment for faster execution
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, defaultdict
import random
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Import maze environment
from intrinsic_motivation_complex_maze import ComplexMazeEnvironment, ImprovedKnowledgeGraph


class SimplifiedComplexAgent:
    """Simplified agent for complex maze navigation"""
    
    def __init__(self, state_size, action_size, 
                 use_ged=True, use_ig=True, 
                 intrinsic_weight=0.2):
        
        self.state_size = state_size
        self.action_size = action_size
        self.use_ged = use_ged
        self.use_ig = use_ig
        self.intrinsic_weight = intrinsic_weight
        
        # Smaller network for faster training
        self.q_network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.memory = deque(maxlen=5000)
        
        # Knowledge graph for GED
        if use_ged:
            self.knowledge_graph = ImprovedKnowledgeGraph(max_nodes=200)
        
        # State visits for IG
        self.state_visits = defaultdict(int)
        self.complexity_history = []
        self.prev_complexity = None
        
        # Exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Stats tracking
        self.intrinsic_rewards = []
        self.ged_values = []
        self.ig_values = []
    
    def _calculate_ig(self, state):
        """Simple information gain"""
        if not self.use_ig:
            return 1.0
        
        # Compress state
        state_key = tuple(state[::20])  # Sample every 20th element
        visits = self.state_visits[state_key]
        self.state_visits[state_key] += 1
        
        # Novelty bonus
        ig = 1.0 / (1.0 + visits) + (0.5 if visits == 0 else 0)
        self.ig_values.append(ig)
        return ig
    
    def _calculate_ged(self, state, next_state, action, reward, info):
        """Simplified GED based on structural learning"""
        if not self.use_ged:
            return 0.0
        
        # Add to knowledge graph
        self.knowledge_graph.add_transition(state, next_state, action, reward, info)
        
        # Calculate complexity change
        current_complexity = len(self.knowledge_graph.graph) + len(self.knowledge_graph.graph.edges())
        
        if self.prev_complexity is None:
            self.prev_complexity = current_complexity
            ged = 0.0
        else:
            # Positive when complexity reduces or stays manageable
            complexity_change = current_complexity - self.prev_complexity
            self.prev_complexity = current_complexity
            
            # Bonus for progress
            progress_bonus = 0
            if 'keys_collected' in info:
                progress_bonus += info['keys_collected'] * 0.2
            if 'subgoals_visited' in info:
                progress_bonus += info['subgoals_visited'] * 0.1
            
            ged = -complexity_change * 0.01 + progress_bonus
        
        self.ged_values.append(ged)
        self.complexity_history.append(current_complexity)
        return ged
    
    def calculate_intrinsic_reward(self, state, action, next_state, reward, info):
        """Calculate intrinsic motivation"""
        if not self.use_ged and not self.use_ig:
            return 0.0
        
        ig = self._calculate_ig(next_state)
        ged = self._calculate_ged(state, next_state, action, reward, info)
        
        # Combine with different strategies
        if self.use_ged and self.use_ig:
            intrinsic = ig * max(0, ged + 0.5)  # Offset GED to be more positive
        elif self.use_ig:
            intrinsic = ig
        else:  # GED only
            intrinsic = max(0, ged)
        
        self.intrinsic_rewards.append(intrinsic)
        return intrinsic
    
    def act(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        with torch.no_grad():
            q_values = self.q_network(torch.FloatTensor(state))
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done, info):
        """Store experience"""
        intrinsic = self.calculate_intrinsic_reward(state, action, next_state, reward, info)
        total_reward = reward + self.intrinsic_weight * intrinsic
        self.memory.append((state, action, total_reward, next_state, done))
    
    def replay(self, batch_size=32):
        """Experience replay"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in batch]))
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor(np.array([e[3] for e in batch]))
        dones = torch.FloatTensor([e[4] for e in batch])
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q = self.q_network(next_states).max(1)[0].detach()
        target_q = rewards + 0.99 * next_q * (1 - dones)
        
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def run_single_experiment(config, episodes=100, verbose=True):
    """Run a single configuration"""
    
    env = ComplexMazeEnvironment(size=12, num_rooms=4, num_keys=2)
    agent = SimplifiedComplexAgent(
        env.state_space_size,
        env.action_space_size,
        use_ged=config['use_ged'],
        use_ig=config['use_ig'],
        intrinsic_weight=config['weight']
    )
    
    results = {
        'episode_rewards': [],
        'episode_success': [],
        'episode_steps': [],
        'keys_collected': [],
        'subgoals_reached': [],
        'intrinsic_rewards': []
    }
    
    start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 300:  # Limit steps per episode
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done, info)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Train every few steps
            if len(agent.memory) > 32 and steps % 4 == 0:
                agent.replay()
        
        # Record episode stats
        results['episode_rewards'].append(total_reward)
        results['episode_success'].append(1 if info.get('success', False) else 0)
        results['episode_steps'].append(steps)
        results['keys_collected'].append(info.get('keys_collected', 0))
        results['subgoals_reached'].append(info.get('subgoals_visited', 0))
        
        if agent.intrinsic_rewards:
            results['intrinsic_rewards'].append(np.mean(agent.intrinsic_rewards[-steps:]))
        
        # Progress report
        if verbose and (episode + 1) % 20 == 0:
            recent_success = np.mean(results['episode_success'][-10:])
            recent_reward = np.mean(results['episode_rewards'][-10:])
            print(f"  Episode {episode+1}: Success={recent_success:.2f}, Reward={recent_reward:.2f}")
    
    # Calculate summary statistics
    summary = {
        'success_rate': np.mean(results['episode_success']),
        'avg_reward': np.mean(results['episode_rewards']),
        'avg_steps': np.mean(results['episode_steps']),
        'avg_keys': np.mean(results['keys_collected']),
        'avg_subgoals': np.mean(results['subgoals_reached']),
        'final_success': np.mean(results['episode_success'][-20:]),
        'convergence_episode': next((i for i, s in enumerate(results['episode_success']) 
                                   if np.mean(results['episode_success'][i:i+10]) >= 0.5), episodes),
        'training_time': time.time() - start_time
    }
    
    # Add agent statistics
    if hasattr(agent, 'complexity_history') and agent.complexity_history:
        summary['final_complexity'] = agent.complexity_history[-1]
        summary['complexity_change'] = agent.complexity_history[-1] - agent.complexity_history[0] if len(agent.complexity_history) > 1 else 0
    
    return results, summary


def main():
    """Run the complex maze experiment"""
    
    output_dir = Path("experiments/foundational_intrinsic_motivation/results_complex_maze")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("COMPLEX MAZE EXPERIMENT - LIGHTWEIGHT VERSION")
    print("="*70)
    
    # Test configurations
    configs = [
        {"name": "Full_GED_IG", "use_ged": True, "use_ig": True, "weight": 0.2},
        {"name": "IG_Only", "use_ged": False, "use_ig": True, "weight": 0.2},
        {"name": "GED_Only", "use_ged": True, "use_ig": False, "weight": 0.2},
        {"name": "Baseline", "use_ged": False, "use_ig": False, "weight": 0.0}
    ]
    
    all_results = {}
    episodes = 100
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        results, summary = run_single_experiment(config, episodes=episodes)
        all_results[config['name']] = {
            'results': results,
            'summary': summary
        }
        
        print(f"Final performance: Success={summary['success_rate']:.3f}, "
              f"Keys={summary['avg_keys']:.2f}, Time={summary['training_time']:.1f}s")
    
    # Save results
    save_data = {
        'configurations': configs,
        'episodes': episodes,
        'results': {
            name: {
                'summary': data['summary'],
                'final_performance': {
                    'success_rate': data['summary']['success_rate'],
                    'convergence': data['summary']['convergence_episode'],
                    'final_success': data['summary']['final_success']
                }
            }
            for name, data in all_results.items()
        },
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'random_seed': RANDOM_SEED,
            'maze_config': {
                'size': 12,
                'rooms': 4,
                'keys': 2
            }
        }
    }
    
    with open(output_dir / 'lightweight_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    
    # Create visualizations
    create_visualizations(all_results, output_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    print("\nFinal Success Rates:")
    for name, data in all_results.items():
        success = data['summary']['success_rate']
        final_success = data['summary']['final_success']
        print(f"  {name}: {success:.3f} (final 20 eps: {final_success:.3f})")
    
    print("\nConvergence Speed:")
    for name, data in all_results.items():
        conv = data['summary']['convergence_episode']
        print(f"  {name}: {conv} episodes" + (" (converged)" if conv < episodes else " (not converged)"))
    
    print("\nProgress Metrics:")
    for name, data in all_results.items():
        keys = data['summary']['avg_keys']
        subgoals = data['summary']['avg_subgoals']
        print(f"  {name}: Keys={keys:.2f}/2, Subgoals={subgoals:.2f}")
    
    print(f"\nResults saved to {output_dir}")


def create_visualizations(all_results, output_dir):
    """Create simple visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Learning curves
    ax = axes[0, 0]
    for name, data in all_results.items():
        rewards = data['results']['episode_rewards']
        # Smooth with moving average
        window = 10
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=name, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Learning Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Success rate over time
    ax = axes[0, 1]
    for name, data in all_results.items():
        success = data['results']['episode_success']
        # Cumulative success rate
        cumsum = np.cumsum(success)
        cumrate = cumsum / (np.arange(len(success)) + 1)
        ax.plot(cumrate, label=name, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Success Rate')
    ax.set_title('Success Rate Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final performance comparison
    ax = axes[1, 0]
    names = list(all_results.keys())
    success_rates = [all_results[n]['summary']['success_rate'] for n in names]
    final_rates = [all_results[n]['summary']['final_success'] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, success_rates, width, label='Overall')
    ax.bar(x + width/2, final_rates, width, label='Final 20 eps')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Success Rate')
    ax.set_title('Performance Comparison')
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Progress metrics
    ax = axes[1, 1]
    keys = [all_results[n]['summary']['avg_keys'] for n in names]
    subgoals = [all_results[n]['summary']['avg_subgoals'] for n in names]
    
    ax.bar(x - width/2, keys, width, label='Keys (of 2)')
    ax.bar(x + width/2, subgoals, width, label='Subgoals')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Average Count')
    ax.set_title('Exploration Progress')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'complex_maze_results.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()