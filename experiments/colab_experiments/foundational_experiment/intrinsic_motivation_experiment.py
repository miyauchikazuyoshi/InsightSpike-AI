#!/usr/bin/env python3
"""
Foundational Experiment: Intrinsic Motivation (ΔGED × ΔIG) Effectiveness
=====================================================================

This experiment quantifies the effectiveness of intrinsic motivation rewards
in simple environments using Grid-World maze and MountainCar.

Experimental Design:
- Environments: Grid-World maze, MountainCar
- Ablation Study: Compare ΔGED=0, ΔIG=0, and full ΔGED×ΔIG conditions
- Metrics: Success rate, episode count, sample efficiency, learning curves
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Environment imports
try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        GYM_AVAILABLE = True
        print("⚠️ Using legacy gym. Consider upgrading to gymnasium.")
    except ImportError:
        GYM_AVAILABLE = False
        print("⚠️ Neither gymnasium nor gym available. MountainCar environment will be skipped.")
        
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# InsightSpike-AI imports (will be available after cloning in Colab)
try:
    from insightspike.algorithms.graph_edit_distance import GraphEditDistance
    from insightspike.algorithms.information_gain import InformationGain
    from insightspike.core.config_manager import ConfigManager
    INSIGHTSPIKE_AVAILABLE = True
except ImportError:
    # Fallback for when InsightSpike-AI is not available
    INSIGHTSPIKE_AVAILABLE = False
    print("⚠️  InsightSpike-AI not available, using simplified implementations")

class SimpleGridWorld:
    """Simple Grid-World environment for testing intrinsic motivation"""
    
    def __init__(self, size=8, num_obstacles=None):
        self.size = size
        self.num_obstacles = num_obstacles or size // 2
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.grid = np.zeros((self.size, self.size))
        
        # Place obstacles randomly
        obstacle_positions = np.random.choice(
            self.size * self.size, 
            self.num_obstacles, 
            replace=False
        )
        for pos in obstacle_positions:
            row, col = pos // self.size, pos % self.size
            if (row, col) != (0, 0) and (row, col) != (self.size-1, self.size-1):
                self.grid[row, col] = -1  # Obstacle
        
        # Set start and goal
        self.start_pos = (0, 0)
        self.goal_pos = (self.size-1, self.size-1)
        self.current_pos = self.start_pos
        self.grid[self.goal_pos] = 1  # Goal
        
        self.step_count = 0
        self.max_steps = self.size * self.size * 2
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation"""
        state = np.zeros((self.size, self.size, 3))  # Position, obstacles, goal
        
        # Current position channel
        state[self.current_pos[0], self.current_pos[1], 0] = 1
        
        # Obstacles channel
        state[:, :, 1] = (self.grid == -1).astype(float)
        
        # Goal channel
        state[self.goal_pos[0], self.goal_pos[1], 2] = 1
        
        return state.flatten()
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        # Actions: 0=up, 1=right, 2=down, 3=left
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        if action < len(moves):
            move = moves[action]
            new_pos = (
                max(0, min(self.size-1, self.current_pos[0] + move[0])),
                max(0, min(self.size-1, self.current_pos[1] + move[1]))
            )
            
            # Check if new position is valid (not an obstacle)
            if self.grid[new_pos] != -1:
                self.current_pos = new_pos
        
        self.step_count += 1
        
        # Calculate reward
        reward = -0.01  # Small step penalty
        done = False
        
        if self.current_pos == self.goal_pos:
            reward = 1.0  # Goal reward
            done = True
        elif self.step_count >= self.max_steps:
            reward = -0.1  # Timeout penalty
            done = True
            
        return self._get_state(), reward, done
    
    @property
    def action_space_size(self):
        return 4
    
    @property  
    def state_space_size(self):
        return self.size * self.size * 3

class IntrinsicMotivationAgent:
    """Agent with intrinsic motivation based on ΔGED × ΔIG"""
    
    def __init__(self, state_size, action_size, use_ged=True, use_ig=True, 
                 learning_rate=0.001, epsilon=1.0, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.use_ged = use_ged
        self.use_ig = use_ig
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        # Initialize InsightSpike-AI components if available
        if INSIGHTSPIKE_AVAILABLE:
            self.ged_calculator = GraphEditDistance() if use_ged else None
            self.ig_calculator = InformationGain() if use_ig else None
            print("✅ Using InsightSpike-AI intrinsic motivation components")
        else:
            self.ged_calculator = None
            self.ig_calculator = None
            print("⚠️  Using simplified intrinsic motivation fallback")
        
        # Neural network for Q-values
        self.q_network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = []
        self.max_memory = 10000
        
        # For intrinsic motivation calculation
        self.previous_states = []
        self.state_visit_counts = {}
        
    def _calculate_ged(self, state1, state2):
        """Calculate Graph Edit Distance (simplified as state difference)"""
        if not self.use_ged:
            return 0.0
            
        if INSIGHTSPIKE_AVAILABLE and self.ged_calculator:
            # Use actual InsightSpike-AI GED calculation
            # Convert states to graph representation (simplified)
            graph1 = self._state_to_graph(state1)
            graph2 = self._state_to_graph(state2)
            return self.ged_calculator.calculate_distance(graph1, graph2)
        else:
            # Fallback: simple L2 distance
            return np.linalg.norm(state1 - state2)
    
    def _calculate_ig(self, state):
        """Calculate Information Gain (simplified as novelty)"""
        if not self.use_ig:
            return 0.0
        
        if INSIGHTSPIKE_AVAILABLE and self.ig_calculator:
            # Use actual InsightSpike-AI IG calculation
            return self.ig_calculator.calculate_gain(state, self.state_visit_counts)
        else:
            # Fallback: frequency-based novelty
            state_key = tuple(state.flatten())
            visit_count = self.state_visit_counts.get(state_key, 0)
            self.state_visit_counts[state_key] = visit_count + 1
            return 1.0 / (1.0 + visit_count)
    
    def _state_to_graph(self, state):
        """Convert state to graph representation for GED calculation"""
        # Simplified graph representation
        # In practice, this would be more sophisticated
        return {
            'nodes': list(range(len(state))),
            'edges': [(i, i+1) for i in range(len(state)-1)],
            'features': state.tolist()
        }
    
    def _calculate_intrinsic_reward(self, current_state):
        """Calculate intrinsic reward as ΔGED × ΔIG"""
        if not self.previous_states:
            self.previous_states.append(current_state)
            return 0.0
        
        # Calculate deltas
        prev_state = self.previous_states[-1]
        delta_ged = self._calculate_ged(current_state, prev_state)
        delta_ig = self._calculate_ig(current_state)
        
        intrinsic_reward = delta_ged * delta_ig
        
        # Update state history
        self.previous_states.append(current_state)
        if len(self.previous_states) > 10:  # Keep only recent states
            self.previous_states.pop(0)
            
        return intrinsic_reward
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        # Add intrinsic reward
        intrinsic_reward = self._calculate_intrinsic_reward(next_state)
        total_reward = reward + 0.1 * intrinsic_reward  # Weight intrinsic reward
        
        experience = (state, action, total_reward, next_state, done)
        self.memory.append(experience)
        
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)
    
    def replay(self, batch_size=32):
        """Train the network on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        batch_experiences = [self.memory[i] for i in batch]
        
        states = torch.FloatTensor([e[0] for e in batch_experiences])
        actions = torch.LongTensor([e[1] for e in batch_experiences])
        rewards = torch.FloatTensor([e[2] for e in batch_experiences])
        next_states = torch.FloatTensor([e[3] for e in batch_experiences])
        dones = torch.BoolTensor([e[4] for e in batch_experiences])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.q_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def run_grid_world_experiment(episodes=500, trials=5):
    """Run Grid-World experiment with different agent configurations"""
    
    configurations = [
        {"name": "Full (ΔGED × ΔIG)", "use_ged": True, "use_ig": True},
        {"name": "No GED (ΔIG only)", "use_ged": False, "use_ig": True},
        {"name": "No IG (ΔGED only)", "use_ged": True, "use_ig": False},
        {"name": "Baseline (No intrinsic)", "use_ged": False, "use_ig": False}
    ]
    
    results = {}
    
    for config in configurations:
        print(f"Running configuration: {config['name']}")
        config_results = {
            "success_rates": [],
            "episode_counts": [],
            "learning_curves": [],
            "sample_efficiency": []
        }
        
        for trial in range(trials):
            print(f"  Trial {trial + 1}/{trials}")
            
            env = SimpleGridWorld(size=8)
            agent = IntrinsicMotivationAgent(
                state_size=env.state_space_size,
                action_size=env.action_space_size,
                use_ged=config["use_ged"],
                use_ig=config["use_ig"]
            )
            
            trial_rewards = []
            successes = 0
            
            for episode in range(episodes):
                state = env.reset()
                total_reward = 0
                steps = 0
                
                while True:
                    action = agent.act(state)
                    next_state, reward, done = env.step(action)
                    agent.remember(state, action, reward, next_state, done)
                    
                    state = next_state
                    total_reward += reward
                    steps += 1
                    
                    if done:
                        if env.current_pos == env.goal_pos:
                            successes += 1
                        break
                
                trial_rewards.append(total_reward)
                
                # Train agent
                if len(agent.memory) > 32:
                    agent.replay()
            
            config_results["success_rates"].append(successes / episodes)
            config_results["episode_counts"].append(episodes)
            config_results["learning_curves"].append(trial_rewards)
            config_results["sample_efficiency"].append(np.mean(trial_rewards[-50:]))  # Last 50 episodes
        
        results[config["name"]] = config_results
    
    return results

def run_mountain_car_experiment(episodes=500, trials=3):
    """Run MountainCar experiment with intrinsic motivation"""
    
    try:
        # Use gymnasium if available, fallback to gym
        if not GYM_AVAILABLE:
            print("Gym/Gymnasium not available, skipping MountainCar experiment")
            return {}
    except ImportError:
        print("Gym not available, skipping MountainCar experiment")
        return {}
    
    configurations = [
        {"name": "Full (ΔGED × ΔIG)", "use_ged": True, "use_ig": True},
        {"name": "No GED (ΔIG only)", "use_ged": False, "use_ig": True},
        {"name": "No IG (ΔGED only)", "use_ged": True, "use_ig": False},
        {"name": "Baseline (No intrinsic)", "use_ged": False, "use_ig": False}
    ]
    
    results = {}
    
    for config in configurations:
        print(f"Running MountainCar configuration: {config['name']}")
        config_results = {
            "success_rates": [],
            "episode_lengths": [],
            "learning_curves": []
        }
        
        for trial in range(trials):
            print(f"  Trial {trial + 1}/{trials}")
            
            env = gym.make('MountainCar-v0')
            agent = IntrinsicMotivationAgent(
                state_size=env.observation_space.shape[0],
                action_size=env.action_space.n,
                use_ged=config["use_ged"],
                use_ig=config["use_ig"]
            )
            
            trial_lengths = []
            successes = 0
            
            for episode in range(episodes):
                state = env.reset()
                if isinstance(state, tuple):
                    state = state[0]  # Handle new gym API
                    
                total_reward = 0
                steps = 0
                
                while steps < 200:  # MountainCar episode limit
                    action = agent.act(state)
                    result = env.step(action)
                    
                    if len(result) == 4:
                        next_state, reward, done, info = result
                    else:
                        next_state, reward, done, truncated, info = result
                        done = done or truncated
                    
                    agent.remember(state, action, reward, next_state, done)
                    
                    state = next_state
                    total_reward += reward
                    steps += 1
                    
                    if done:
                        if steps < 200:  # Success if solved before time limit
                            successes += 1
                        break
                
                trial_lengths.append(steps)
                
                # Train agent
                if len(agent.memory) > 32:
                    agent.replay()
            
            config_results["success_rates"].append(successes / episodes)
            config_results["episode_lengths"].append(np.mean(trial_lengths))
            config_results["learning_curves"].append(trial_lengths)
        
        results[config["name"]] = config_results
        env.close()
    
    return results

def create_visualization(grid_results, mountain_car_results=None):
    """Create comprehensive visualization of experimental results"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    if mountain_car_results:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Intrinsic Motivation Effectiveness: Grid-World & MountainCar', fontsize=16)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Intrinsic Motivation Effectiveness: Grid-World Environment', fontsize=16)
        axes = axes.flatten()
    
    # Grid-World Results
    configs = list(grid_results.keys())
    
    # 1. Success Rates Comparison
    ax1 = axes[0] if mountain_car_results else axes[0]
    success_data = []
    for config in configs:
        for rate in grid_results[config]["success_rates"]:
            success_data.append({"Configuration": config, "Success Rate": rate})
    
    success_df = pd.DataFrame(success_data)
    sns.boxplot(data=success_df, x="Configuration", y="Success Rate", ax=ax1)
    ax1.set_title("Grid-World: Success Rates")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    
    # 2. Sample Efficiency
    ax2 = axes[1] if mountain_car_results else axes[1]
    efficiency_data = []
    for config in configs:
        for eff in grid_results[config]["sample_efficiency"]:
            efficiency_data.append({"Configuration": config, "Sample Efficiency": eff})
    
    efficiency_df = pd.DataFrame(efficiency_data)
    sns.boxplot(data=efficiency_df, x="Configuration", y="Sample Efficiency", ax=ax2)
    ax2.set_title("Grid-World: Sample Efficiency (Last 50 Episodes)")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    
    # 3. Learning Curves
    ax3 = axes[2] if mountain_car_results else axes[2]
    for config in configs:
        learning_curves = grid_results[config]["learning_curves"]
        if learning_curves:
            # Average across trials
            max_length = max(len(curve) for curve in learning_curves)
            padded_curves = []
            for curve in learning_curves:
                padded = curve + [curve[-1]] * (max_length - len(curve))
                padded_curves.append(padded)
            
            mean_curve = np.mean(padded_curves, axis=0)
            std_curve = np.std(padded_curves, axis=0)
            
            episodes = range(len(mean_curve))
            ax3.plot(episodes, mean_curve, label=config, linewidth=2)
            ax3.fill_between(episodes, 
                           mean_curve - std_curve, 
                           mean_curve + std_curve, 
                           alpha=0.3)
    
    ax3.set_title("Grid-World: Learning Curves")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Average Reward")
    ax3.legend()
    
    # Statistical Summary
    ax4 = axes[3] if mountain_car_results else axes[3]
    summary_data = []
    for config in configs:
        mean_success = np.mean(grid_results[config]["success_rates"])
        std_success = np.std(grid_results[config]["success_rates"])
        mean_efficiency = np.mean(grid_results[config]["sample_efficiency"])
        
        summary_data.append({
            "Configuration": config,
            "Mean Success Rate": mean_success,
            "Std Success Rate": std_success,
            "Mean Sample Efficiency": mean_efficiency
        })
    
    summary_df = pd.DataFrame(summary_data)
    ax4.axis('tight')
    ax4.axis('off')
    
    table = ax4.table(cellText=summary_df.round(3).values,
                     colLabels=summary_df.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax4.set_title("Statistical Summary")
    
    # MountainCar Results (if available)
    if mountain_car_results:
        # 5. MountainCar Success Rates
        ax5 = axes[4]
        mc_success_data = []
        for config in mountain_car_results.keys():
            for rate in mountain_car_results[config]["success_rates"]:
                mc_success_data.append({"Configuration": config, "Success Rate": rate})
        
        if mc_success_data:
            mc_success_df = pd.DataFrame(mc_success_data)
            sns.boxplot(data=mc_success_df, x="Configuration", y="Success Rate", ax=ax5)
            ax5.set_title("MountainCar: Success Rates")
            ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
        
        # 6. MountainCar Episode Lengths
        ax6 = axes[5]
        mc_length_data = []
        for config in mountain_car_results.keys():
            for length in mountain_car_results[config]["episode_lengths"]:
                mc_length_data.append({"Configuration": config, "Episode Length": length})
        
        if mc_length_data:
            mc_length_df = pd.DataFrame(mc_length_data)
            sns.boxplot(data=mc_length_df, x="Configuration", y="Episode Length", ax=ax6)
            ax6.set_title("MountainCar: Average Episode Lengths")
            ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    return fig

def run_foundational_experiment():
    """Run the complete foundational experiment"""
    
    print("🚀 Starting Foundational Experiment: Intrinsic Motivation Effectiveness")
    print("=" * 70)
    
    # Run Grid-World experiment
    print("\n📊 Running Grid-World Experiment...")
    grid_results = run_grid_world_experiment(episodes=300, trials=3)
    
    # Run MountainCar experiment (optional)
    print("\n🏔️ Running MountainCar Experiment...")
    try:
        mountain_car_results = run_mountain_car_experiment(episodes=200, trials=2)
    except Exception as e:
        print(f"MountainCar experiment failed: {e}")
        mountain_car_results = None
    
    # Create visualizations
    print("\n📈 Creating Visualizations...")
    fig = create_visualization(grid_results, mountain_car_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("foundational_experiment_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save plot
    plot_path = results_dir / f"intrinsic_motivation_results_{timestamp}.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved: {plot_path}")
    
    # Save data
    results_data = {
        "timestamp": timestamp,
        "grid_world": grid_results,
        "mountain_car": mountain_car_results,
        "experimental_setup": {
            "grid_world_episodes": 300,
            "grid_world_trials": 3,
            "mountain_car_episodes": 200,
            "mountain_car_trials": 2
        }
    }
    
    data_path = results_dir / f"experimental_data_{timestamp}.json"
    with open(data_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        json.dump(convert_numpy(results_data), f, indent=2)
    
    print(f"💾 Experimental data saved: {data_path}")
    
    # Print summary
    print("\n📋 Experimental Summary:")
    print("-" * 50)
    
    for config in grid_results.keys():
        mean_success = np.mean(grid_results[config]["success_rates"])
        std_success = np.std(grid_results[config]["success_rates"])
        print(f"{config:25} | Success Rate: {mean_success:.3f} ± {std_success:.3f}")
    
    print("\n✅ Foundational Experiment Complete!")
    
    return {
        "results": results_data,
        "visualization": fig,
        "paths": {
            "plot": plot_path,
            "data": data_path
        }
    }

def setup_foundational_experiment_directories(base_dir="data/foundational_experiments"):
    """Setup directory structure for foundational experiments"""
    from pathlib import Path
    
    # Create experiment directories
    dirs = {
        "base": Path(base_dir),
        "environments": Path(base_dir) / "environments",
        "baselines": Path(base_dir) / "baselines", 
        "results": Path(base_dir) / "results",
        "models": Path(base_dir) / "models",
        "logs": Path(base_dir) / "logs"
    }
    
    for name, dir_path in dirs.items():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create environment-specific directories
    environments = ["gridworld", "mountaincar"]
    for env in environments:
        env_dir = dirs["environments"] / env
        env_dir.mkdir(exist_ok=True)
        
        # Create subdirs for each environment
        for subdir in ["configs", "results", "visualizations"]:
            (env_dir / subdir).mkdir(exist_ok=True)
    
    # Create baseline agent directories
    baselines = ["random", "greedy", "q_learning", "intrinsic_full", "intrinsic_ged_only", "intrinsic_ig_only"]
    for baseline in baselines:
        baseline_dir = dirs["baselines"] / baseline
        baseline_dir.mkdir(exist_ok=True)
        
        for subdir in ["models", "results", "logs"]:
            (baseline_dir / subdir).mkdir(exist_ok=True)
    
    return dirs

def save_foundational_experiment_results(results: Dict, experiment_dirs: Dict, experiment_name: str = "intrinsic_motivation"):
    """Save foundational experiment results with comprehensive data management and environment tracking"""
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("📊 Collecting environment information for reproducibility...")
    env_info = collect_environment_info()
    
    # Create comprehensive results structure
    comprehensive_results = {
        "experiment_metadata": {
            "name": experiment_name,
            "timestamp": timestamp,
            "environments_tested": list(results.keys()),
            "configurations": {
                "Full (ΔGED × ΔIG)": {"use_ged": True, "use_ig": True},
                "No GED (ΔIG only)": {"use_ged": False, "use_ig": True},
                "No IG (ΔGED only)": {"use_ged": True, "use_ig": False},
                "Baseline (No intrinsic)": {"use_ged": False, "use_ig": False}
            },
            "experiment_id": f"{experiment_name}_{timestamp}",
            "reproducibility_notes": {
                "random_seed": "Recommend setting torch.manual_seed() for reproducibility",
                "deterministic_operations": "Consider torch.backends.cudnn.deterministic = True"
            }
        },
        "environment_info": env_info,
        "results": results,
        "statistical_analysis": {},
        "performance_summary": {}
    }
    
    # Calculate statistical analysis for each environment
    for env_name, env_results in results.items():
        if isinstance(env_results, dict) and "configurations" in env_results:
            configs = env_results["configurations"]
            
            # Performance summary per configuration
            comprehensive_results["performance_summary"][env_name] = {}
            
            for config_name, config_data in configs.items():
                if "trials" in config_data:
                    trials = config_data["trials"]
                    
                    # Calculate summary statistics
                    success_rates = [trial.get("success_rate", 0) for trial in trials]
                    episode_counts = [trial.get("episodes", 0) for trial in trials] 
                    final_rewards = [trial.get("final_reward", 0) for trial in trials]
                    
                    comprehensive_results["performance_summary"][env_name][config_name] = {
                        "success_rate": {
                            "mean": np.mean(success_rates),
                            "std": np.std(success_rates),
                            "min": np.min(success_rates),
                            "max": np.max(success_rates)
                        },
                        "episodes_to_completion": {
                            "mean": np.mean(episode_counts),
                            "std": np.std(episode_counts),
                            "median": np.median(episode_counts)
                        },
                        "final_reward": {
                            "mean": np.mean(final_rewards),
                            "std": np.std(final_rewards)
                        },
                        "sample_efficiency": np.mean(success_rates) / max(np.mean(episode_counts), 1),
                        "total_trials": len(trials)
                    }
    
    # Save main results file
    results_file = experiment_dirs["results"] / f"{experiment_name}_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    print(f"💾 Saved comprehensive results to {results_file}")
    
    # Save CSV for easy analysis
    csv_file = experiment_dirs["results"] / f"{experiment_name}_{timestamp}.csv"
    save_foundational_results_as_csv(comprehensive_results, csv_file)
    
    return results_file

def save_foundational_results_as_csv(results: Dict, csv_file: Path):
    """Convert foundational experiment results to CSV format"""
    csv_data = []
    
    if "performance_summary" in results:
        for env_name, env_data in results["performance_summary"].items():
            for config_name, config_data in env_data.items():
                row = {
                    "environment": env_name,
                    "configuration": config_name,
                    "success_rate_mean": config_data.get("success_rate", {}).get("mean", 0),
                    "success_rate_std": config_data.get("success_rate", {}).get("std", 0),
                    "episodes_mean": config_data.get("episodes_to_completion", {}).get("mean", 0),
                    "episodes_std": config_data.get("episodes_to_completion", {}).get("std", 0),
                    "final_reward_mean": config_data.get("final_reward", {}).get("mean", 0),
                    "sample_efficiency": config_data.get("sample_efficiency", 0),
                    "total_trials": config_data.get("total_trials", 0)
                }
                csv_data.append(row)
    
    # Save to CSV
    import pandas as pd
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    print(f"📊 Saved CSV summary to {csv_file}")

def collect_environment_info():
    """Collect comprehensive environment and library version information for foundational experiment"""
    import sys
    import platform
    import subprocess
    import pkg_resources
    import os
    
    env_info = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "platform_details": platform.platform()
        },
        "environment_variables": {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "HOME": os.environ.get("HOME", ""),
            "PWD": os.environ.get("PWD", "")
        },
        "libraries": {},
        "hardware_info": {}
    }
    
    # Collect library versions for RL experiment
    important_libraries = [
        'numpy', 'matplotlib', 'seaborn', 'pandas', 'torch', 'gymnasium',
        'gym', 'plotly', 'jupyter', 'ipython'
    ]
    
    for lib in important_libraries:
        try:
            version = pkg_resources.get_distribution(lib).version
            env_info["libraries"][lib] = version
        except pkg_resources.DistributionNotFound:
            env_info["libraries"][lib] = "not_installed"
        except Exception as e:
            env_info["libraries"][lib] = f"error: {str(e)}"
    
    # PyTorch specific info for RL experiments
    try:
        import torch
        env_info["pytorch_info"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "random_seed_set": False,  # Will be updated if seed is set
            "deterministic_mode": torch.backends.cudnn.deterministic if torch.cuda.is_available() else False
        }
    except ImportError:
        env_info["pytorch_info"] = {"error": "PyTorch not available"}
    
    # Gymnasium/Gym environment info
    try:
        import gymnasium as gym
        env_info["rl_environment"] = {
            "library": "gymnasium",
            "version": gym.__version__
        }
    except ImportError:
        try:
            import gym
            env_info["rl_environment"] = {
                "library": "gym",
                "version": gym.__version__,
                "note": "Using legacy gym library"
            }
        except ImportError:
            env_info["rl_environment"] = {"error": "No RL environment library available"}
    
    # Memory and CPU info
    try:
        import psutil
        env_info["hardware_info"]["memory"] = {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "percent_used": psutil.virtual_memory().percent
        }
        env_info["hardware_info"]["cpu"] = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        }
    except ImportError:
        env_info["hardware_info"]["note"] = "psutil not available for detailed hardware info"
    
    # Git information
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                         stderr=subprocess.DEVNULL).decode('ascii').strip()
        git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                                           stderr=subprocess.DEVNULL).decode('ascii').strip()
        env_info["git_info"] = {
            "commit_hash": git_hash,
            "branch": git_branch
        }
        
        # Check for uncommitted changes
        try:
            subprocess.check_output(['git', 'diff', '--quiet'])
            env_info["git_info"]["clean_working_directory"] = True
        except subprocess.CalledProcessError:
            env_info["git_info"]["clean_working_directory"] = False
            env_info["git_info"]["warning"] = "Uncommitted changes detected"
    except (subprocess.CalledProcessError, FileNotFoundError):
        env_info["git_info"] = {"error": "Git not available or not in git repository"}
    
    # Execution environment detection
    env_info["execution_environment"] = "unknown"
    if "google.colab" in sys.modules:
        env_info["execution_environment"] = "google_colab"
    elif "ipykernel" in sys.modules:
        env_info["execution_environment"] = "jupyter"
    elif hasattr(sys, 'ps1'):
        env_info["execution_environment"] = "interactive_python"
    else:
        env_info["execution_environment"] = "script"
    
    return env_info
