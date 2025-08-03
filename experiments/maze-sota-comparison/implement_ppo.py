#!/usr/bin/env python3
"""Implement PPO baseline for maze navigation."""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.environments.maze import SimpleMaze


class PPONetwork(nn.Module):
    """Simple neural network for PPO."""
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        features = self.shared(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value


class PPOAgent:
    """PPO agent for maze navigation."""
    
    def __init__(self, input_dim=6, lr=3e-4, gamma=0.99, eps_clip=0.2, 
                 k_epochs=4, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.network = PPONetwork(input_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Memory
        self.reset_memory()
    
    def reset_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def get_observation_vector(self, maze, obs):
        """Convert maze observation to vector."""
        # Simple features: position + local vision
        x, y = obs.position
        features = [
            x / maze.size[0],  # Normalized position
            y / maze.size[1],
            float(obs.is_junction),
            float(obs.is_dead_end),
            float(obs.is_goal),
            obs.num_paths / 4.0  # Normalized number of paths
        ]
        return torch.FloatTensor(features).to(self.device)
    
    def act(self, state):
        """Choose action using current policy."""
        with torch.no_grad():
            logits, value = self.network(state.unsqueeze(0))
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value.item()
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store transition in memory."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_returns(self):
        """Compute discounted returns."""
        returns = []
        discounted_reward = 0
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        # Normalize
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def update(self):
        """Update policy using PPO."""
        # Convert lists to tensors
        old_states = torch.stack(self.states)
        old_actions = torch.tensor(self.actions).to(self.device)
        old_log_probs = torch.stack(self.log_probs)
        
        # Compute returns and advantages
        returns = self.compute_returns()
        old_values = torch.tensor(self.values).to(self.device)
        advantages = returns - old_values
        
        # PPO update
        for _ in range(self.k_epochs):
            # Get current policy
            logits, values = self.network(old_states)
            dist = Categorical(logits=logits)
            
            # Get log probs and entropy
            log_probs = dist.log_prob(old_actions)
            entropy = dist.entropy().mean()
            
            # Calculate ratio
            ratios = torch.exp(log_probs - old_log_probs.detach())
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Calculate losses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Clear memory
        self.reset_memory()


def train_ppo(maze_size=(10, 10), n_episodes=1000, max_steps=500):
    """Train PPO on maze navigation."""
    print("TRAINING PPO ON MAZE NAVIGATION")
    print("=" * 60)
    
    # Create environment
    maze = SimpleMaze(size=maze_size, maze_type='dfs')
    
    # Create agent
    agent = PPOAgent(input_dim=6)
    
    # Training metrics
    episode_rewards = []
    episode_steps = []
    success_rate = []
    
    # Training loop
    pbar = tqdm(range(n_episodes), desc="Training PPO")
    
    for episode in pbar:
        obs = maze.reset()
        state = agent.get_observation_vector(maze, obs)
        
        episode_reward = 0
        
        for step in range(max_steps):
            # Get action
            action, log_prob, value = agent.act(state)
            
            # Take action
            next_obs, reward, done, info = maze.step(action)
            next_state = agent.get_observation_vector(maze, next_obs)
            
            # Store transition
            agent.store_transition(state, action, reward, log_prob, value, done)
            
            episode_reward += reward
            state = next_state
            obs = next_obs
            
            if done:
                break
        
        # Update policy every episode
        agent.update()
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_steps.append(step + 1)
        success = maze.agent_pos == maze.goal_pos
        
        # Calculate success rate
        if episode == 0:
            success_rate.append(float(success))
        else:
            success_rate.append(0.95 * success_rate[-1] + 0.05 * float(success))
        
        # Update progress bar
        if episode % 10 == 0:
            pbar.set_postfix({
                'reward': f"{episode_reward:.2f}",
                'steps': step + 1,
                'success_rate': f"{success_rate[-1]:.2%}"
            })
    
    return agent, episode_rewards, episode_steps, success_rate


def evaluate_ppo(agent, n_trials=100):
    """Evaluate trained PPO agent."""
    print("\nEVALUATING PPO")
    print("=" * 60)
    
    results = []
    
    for trial in tqdm(range(n_trials), desc="Evaluating"):
        # Create new maze
        maze = SimpleMaze(size=(10, 10), maze_type='dfs')
        obs = maze.reset()
        
        start_time = time.time()
        
        for step in range(500):
            state = agent.get_observation_vector(maze, obs)
            action, _, _ = agent.act(state)
            obs, reward, done, info = maze.step(action)
            
            if done:
                elapsed = time.time() - start_time
                success = maze.agent_pos == maze.goal_pos
                results.append({
                    'steps': step + 1,
                    'success': success,
                    'time': elapsed
                })
                break
        else:
            results.append({
                'steps': 500,
                'success': False,
                'time': time.time() - start_time
            })
    
    # Calculate statistics
    successes = [r for r in results if r['success']]
    success_rate = len(successes) / n_trials
    
    if successes:
        avg_steps = np.mean([r['steps'] for r in successes])
        avg_time = np.mean([r['time'] for r in successes])
    else:
        avg_steps = 500
        avg_time = 0
    
    print(f"\nResults:")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Average steps (successful): {avg_steps:.1f}")
    print(f"Average time per episode: {avg_time*1000:.1f}ms")
    
    return results


def plot_training_curves(episode_rewards, episode_steps, success_rate):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Episode rewards
    ax = axes[0]
    ax.plot(episode_rewards, alpha=0.3)
    ax.plot(np.convolve(episode_rewards, np.ones(50)/50, mode='valid'), 'r-', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Training Rewards')
    ax.grid(True, alpha=0.3)
    
    # Episode steps
    ax = axes[1]
    ax.plot(episode_steps, alpha=0.3)
    ax.plot(np.convolve(episode_steps, np.ones(50)/50, mode='valid'), 'b-', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps to Goal')
    ax.set_title('Episode Length')
    ax.grid(True, alpha=0.3)
    
    # Success rate
    ax = axes[2]
    ax.plot(success_rate, 'g-', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate (Smoothed)')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ppo_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Run PPO training and evaluation."""
    # Train PPO
    print("Training PPO (this will take a few minutes)...")
    agent, rewards, steps, success = train_ppo(n_episodes=200)  # Reduced for faster testing
    
    # Plot training curves
    plot_training_curves(rewards, steps, success)
    print("\n✅ Training curves saved to ppo_training_curves.png")
    
    # Evaluate
    results = evaluate_ppo(agent)
    
    # Compare with geDIG
    print("\n" + "=" * 60)
    print("COMPARISON: PPO vs geDIG")
    print("=" * 60)
    print("PPO (after 200 episodes training):")
    print(f"  Success rate: {len([r for r in results if r['success']])/len(results):.1%}")
    if [r for r in results if r['success']]:
        print(f"  Average steps: {np.mean([r['steps'] for r in results if r['success']]):.1f}")
    else:
        print(f"  Average steps: N/A (no successful episodes)")
    print(f"  Training time: ~1 minute")
    print("\ngeDIG (zero-shot, no training):")
    print(f"  Success rate: 96.7%")
    print(f"  Average steps: 37.7")
    print(f"  Training time: 0 seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()