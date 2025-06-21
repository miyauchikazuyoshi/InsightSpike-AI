"""
Baseline Comparison Framework for InsightSpike-AI
================================================

Rigorous experimental framework for comparing InsightSpike-AI against baseline methods.
Includes statistical analysis, reproducibility controls, and comprehensive metrics.
"""

import json
import logging
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for reproducible experiments"""
    experiment_name: str
    num_trials: int = 30  # Sufficient for statistical significance
    num_episodes: int = 100
    maze_size: int = 10
    wall_density: float = 0.25
    random_seed: int = 42
    
    # Agent-specific parameters
    learning_rate: float = 0.15
    exploration_rate: float = 0.4
    exploration_decay: float = 0.995
    
    # InsightSpike-specific parameters
    dged_threshold: float = -0.3
    dig_threshold: float = 1.0
    insight_reward_scale: float = 0.0  # Will be varied in experiments
    
    # Statistical parameters
    confidence_level: float = 0.95
    significance_threshold: float = 0.05


@dataclass
class TrialResult:
    """Results from a single trial"""
    trial_id: int
    agent_type: str
    config: ExperimentConfig
    
    # Performance metrics
    episode_rewards: List[float]
    episode_steps: List[int]
    success_rates: List[float]  # Success rate over sliding window
    
    # InsightSpike-specific metrics
    total_insights: int
    insight_episodes: List[int]  # Episodes where insights occurred
    insight_rewards: List[float]  # Rewards when insights occurred
    
    # Learning efficiency metrics
    time_to_first_success: Optional[int]  # Episode number of first success
    average_steps_to_goal: float
    learning_curve_auc: float  # Area under learning curve
    
    # Exploration metrics
    states_visited: int
    exploration_efficiency: float
    
    # Statistical summary
    mean_reward: float
    std_reward: float
    mean_steps: float
    std_steps: float
    final_success_rate: float


class BaselineAgent:
    """Standard Q-learning agent without insight detection"""
    
    def __init__(self, config: ExperimentConfig, agent_id: str = "Baseline"):
        self.config = config
        self.agent_id = agent_id
        
        # Set seeds for reproducibility
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)
        
        # Q-learning parameters
        self.learning_rate = config.learning_rate
        self.exploration_rate = config.exploration_rate
        self.exploration_decay = config.exploration_decay
        self.gamma = 0.95
        
        # State-action value function
        self.q_table = defaultdict(lambda: np.zeros(4))
        
        # Tracking
        self.episode_rewards = []
        self.episode_steps = []
        self.visited_states = set()
        
    def select_action(self, state: Tuple[int, int]) -> int:
        """Epsilon-greedy action selection"""
        if np.random.random() < self.exploration_rate:
            return np.random.randint(4)
        return np.argmax(self.q_table[state])
    
    def update(self, state: Tuple[int, int], action: int, reward: float, 
               next_state: Tuple[int, int], done: bool):
        """Q-learning update"""
        current_q = self.q_table[state][action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        
        self.q_table[state][action] += self.learning_rate * (target_q - current_q)
        
        # Track exploration
        self.visited_states.add(state)
    
    def train_episode(self, environment) -> Dict[str, Any]:
        """Train for one episode"""
        state = environment.reset()
        episode_reward = 0
        episode_steps = 0
        
        while True:
            action = self.select_action(state)
            next_state, reward, done, info = environment.step(action)
            
            self.update(state, action, reward, next_state, done)
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if done or episode_steps > 1000:  # Timeout
                break
        
        # Decay exploration
        self.exploration_rate = max(0.05, self.exploration_rate * self.exploration_decay)
        
        self.episode_rewards.append(episode_reward)
        self.episode_steps.append(episode_steps)
        
        return {
            'reward': episode_reward,
            'steps': episode_steps,
            'success': done and episode_steps <= 1000,
            'insights': 0  # Baseline has no insights
        }


class InsightSpikeAgent:
    """InsightSpike agent with configurable insight reward integration"""
    
    def __init__(self, config: ExperimentConfig, agent_id: str = "InsightSpike"):
        self.config = config
        self.agent_id = agent_id
        
        # Set seeds for reproducibility
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)
        
        # Q-learning parameters (same as baseline for fair comparison)
        self.learning_rate = config.learning_rate
        self.exploration_rate = config.exploration_rate
        self.exploration_decay = config.exploration_decay
        self.gamma = 0.95
        
        # Insight parameters
        self.dged_threshold = config.dged_threshold
        self.dig_threshold = config.dig_threshold
        self.insight_reward_scale = config.insight_reward_scale
        
        # State-action value function
        self.q_table = defaultdict(lambda: np.zeros(4))
        
        # Insight tracking
        self.insight_history = []
        self.exploration_history = []
        self.reward_history = []
        self.state_visit_count = defaultdict(int)
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_steps = []
        self.visited_states = set()
        
        # Insight-enhanced learning
        self.insight_boost_duration = 0
        self.base_learning_rate = self.learning_rate
    
    def calculate_dged(self) -> float:
        """Calculate Δ Global Exploration Difficulty"""
        if len(self.reward_history) < 5:
            return 0.0
        
        recent_efficiency = np.mean(self.reward_history[-5:])
        overall_efficiency = np.mean(self.reward_history)
        
        return recent_efficiency - overall_efficiency
    
    def calculate_dig(self, state: Tuple[int, int], reward: float) -> float:
        """Calculate Δ Information Gain"""
        base_gain = max(reward, 0.1)
        
        # Exploration factor
        unique_states = len(set(self.exploration_history))
        total_states = len(self.exploration_history)
        exploration_factor = unique_states / max(total_states, 1)
        
        # Novelty factor
        visit_count = self.state_visit_count[state]
        novelty_factor = 1.0 / (1.0 + visit_count)
        
        return base_gain * exploration_factor * (1 + novelty_factor)
    
    def detect_insight(self, state: Tuple[int, int], action: int, 
                      reward: float, episode: int, step: int) -> Optional[Dict]:
        """Detect insight moments"""
        dged = self.calculate_dged()
        dig = self.calculate_dig(state, reward)
        
        # Insight conditions
        insight_detected = False
        insight_type = ""
        description = ""
        
        # Primary condition: efficiency drop + high info gain
        if dged < self.dged_threshold and dig > self.dig_threshold:
            insight_detected = True
            insight_type = "strategic_breakthrough"
            description = f"Strategic breakthrough: ΔGED={dged:.3f}, ΔIG={dig:.3f}"
        
        # Goal discovery condition
        elif reward > 50:
            insight_detected = True
            insight_type = "goal_discovery"
            description = f"Goal discovery: reward={reward:.1f}"
        
        # High information gain alone
        elif dig > self.dig_threshold * 1.5:
            insight_detected = True
            insight_type = "exploration_insight"
            description = f"Exploration insight: ΔIG={dig:.3f}"
        
        if insight_detected:
            insight = {
                'episode': episode,
                'step': step,
                'dged': dged,
                'dig': dig,
                'type': insight_type,
                'description': description,
                'state': state,
                'action': action,
                'reward': reward
            }
            
            self.insight_history.append(insight)
            
            # Boost learning temporarily
            self.insight_boost_duration = 10
            
            return insight
        
        return None
    
    def select_action(self, state: Tuple[int, int]) -> int:
        """Insight-enhanced action selection"""
        # Boost exploration after insights
        effective_exploration = self.exploration_rate
        if self.insight_boost_duration > 0:
            effective_exploration = min(self.exploration_rate * 1.5, 0.8)
        
        if np.random.random() < effective_exploration:
            return np.random.randint(4)
        return np.argmax(self.q_table[state])
    
    def update(self, state: Tuple[int, int], action: int, reward: float,
               next_state: Tuple[int, int], done: bool, episode: int, step: int):
        """Q-learning update with insight detection and reward integration"""
        
        # Detect insights
        insight = self.detect_insight(state, action, reward, episode, step)
        
        # Calculate intrinsic reward
        intrinsic_reward = 0.0
        if insight:
            intrinsic_reward = self.insight_reward_scale * insight['dig']
        
        # Total reward (extrinsic + intrinsic)
        total_reward = reward + intrinsic_reward
        
        # Q-learning update with potentially enhanced learning rate
        current_q = self.q_table[state][action]
        
        if done:
            target_q = total_reward
        else:
            target_q = total_reward + self.gamma * np.max(self.q_table[next_state])
        
        # Insight-enhanced learning rate
        effective_lr = self.learning_rate
        if self.insight_boost_duration > 0:
            effective_lr = min(self.base_learning_rate * 1.5, 0.5)
            self.insight_boost_duration -= 1
        
        self.q_table[state][action] += effective_lr * (target_q - current_q)
        
        # Update tracking
        self.exploration_history.append(state)
        self.reward_history.append(reward)
        self.state_visit_count[state] += 1
        self.visited_states.add(state)
    
    def train_episode(self, environment) -> Dict[str, Any]:
        """Train for one episode"""
        state = environment.reset()
        episode_reward = 0
        episode_steps = 0
        insights_this_episode = 0
        
        while True:
            action = self.select_action(state)
            next_state, reward, done, info = environment.step(action)
            
            insights_before = len(self.insight_history)
            self.update(state, action, reward, next_state, done, 
                       len(self.episode_rewards), episode_steps)
            insights_after = len(self.insight_history)
            
            insights_this_episode += (insights_after - insights_before)
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if done or episode_steps > 1000:  # Timeout
                break
        
        # Decay exploration
        self.exploration_rate = max(0.05, self.exploration_rate * self.exploration_decay)
        
        self.episode_rewards.append(episode_reward)
        self.episode_steps.append(episode_steps)
        
        return {
            'reward': episode_reward,
            'steps': episode_steps,
            'success': done and episode_steps <= 1000,
            'insights': insights_this_episode
        }


class SimpleEnvironment:
    """Simple maze environment for controlled experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.size = config.maze_size
        self.wall_density = config.wall_density
        
        # Set seed for reproducible maze generation
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)
        
        self.maze = self._generate_maze()
        self.start = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        self.current_pos = self.start
        
    def _generate_maze(self) -> np.ndarray:
        """Generate reproducible maze"""
        maze = np.zeros((self.size, self.size))
        
        # Add walls with fixed seed
        for i in range(1, self.size - 1):
            for j in range(1, self.size - 1):
                if np.random.random() < self.wall_density:
                    maze[i, j] = 1
        
        # Ensure start and goal are clear
        maze[0, 0] = 0
        maze[self.size - 1, self.size - 1] = 0
        
        return maze
    
    def reset(self) -> Tuple[int, int]:
        """Reset environment"""
        self.current_pos = self.start
        return self.current_pos
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """Execute action"""
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
        dx, dy = moves[action]
        
        new_pos = (self.current_pos[0] + dx, self.current_pos[1] + dy)
        
        # Check bounds and walls
        if (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size and
            self.maze[new_pos[0], new_pos[1]] == 0):
            self.current_pos = new_pos
            reward = -0.01  # Small step penalty
            
            # Goal reward
            if new_pos == self.goal:
                reward = 100.0
                done = True
            else:
                done = False
        else:
            # Wall collision
            reward = -1.0
            done = False
        
        info = {
            'distance_to_goal': abs(self.current_pos[0] - self.goal[0]) + 
                               abs(self.current_pos[1] - self.goal[1])
        }
        
        return self.current_pos, reward, done, info


class ExperimentRunner:
    """Run controlled experiments with statistical analysis"""
    
    def __init__(self, output_dir: str = "experiment_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'experiment.log'),
                logging.StreamHandler()
            ]
        )
    
    def run_single_trial(self, agent_class, config: ExperimentConfig, 
                        trial_id: int) -> TrialResult:
        """Run a single trial with specified agent"""
        
        # Set seeds for this trial
        trial_seed = config.random_seed + trial_id
        np.random.seed(trial_seed)
        random.seed(trial_seed)
        
        # Create environment and agent
        env = SimpleEnvironment(config)
        agent = agent_class(config)
        
        # Training metrics
        episode_rewards = []
        episode_steps = []
        success_rates = []
        insight_episodes = []
        insight_rewards = []
        
        # Track first success
        time_to_first_success = None
        
        logger.info(f"Running trial {trial_id} with {agent.agent_id}")
        
        for episode in range(config.num_episodes):
            result = agent.train_episode(env)
            
            episode_rewards.append(result['reward'])
            episode_steps.append(result['steps'])
            
            # Track insights
            if result['insights'] > 0:
                insight_episodes.append(episode)
                insight_rewards.append(result['reward'])
            
            # Track first success
            if result['success'] and time_to_first_success is None:
                time_to_first_success = episode
            
            # Calculate rolling success rate
            window_size = min(10, episode + 1)
            recent_results = episode_rewards[-window_size:]
            success_rate = sum(1 for r in recent_results if r > 50) / len(recent_results)
            success_rates.append(success_rate)
        
        # Calculate summary statistics
        successful_episodes = [r for r in episode_rewards if r > 50]
        avg_steps_to_goal = np.mean([s for i, s in enumerate(episode_steps) 
                                   if episode_rewards[i] > 50]) if successful_episodes else float('inf')
        
        # Learning curve AUC (area under curve)
        learning_curve_auc = np.trapz(success_rates)
        
        return TrialResult(
            trial_id=trial_id,
            agent_type=agent.agent_id,
            config=config,
            episode_rewards=episode_rewards,
            episode_steps=episode_steps,
            success_rates=success_rates,
            total_insights=getattr(agent, 'insight_history', []),
            insight_episodes=insight_episodes,
            insight_rewards=insight_rewards,
            time_to_first_success=time_to_first_success,
            average_steps_to_goal=avg_steps_to_goal,
            learning_curve_auc=learning_curve_auc,
            states_visited=len(getattr(agent, 'visited_states', set())),
            exploration_efficiency=len(getattr(agent, 'visited_states', set())) / (config.maze_size ** 2),
            mean_reward=np.mean(episode_rewards),
            std_reward=np.std(episode_rewards),
            mean_steps=np.mean(episode_steps),
            std_steps=np.std(episode_steps),
            final_success_rate=success_rates[-1] if success_rates else 0.0
        )
    
    def run_comparison_experiment(self, configs: List[ExperimentConfig]) -> Dict[str, List[TrialResult]]:
        """Run comparison experiment with multiple configurations"""
        
        results = {}
        
        for config in configs:
            logger.info(f"Running experiment: {config.experiment_name}")
            
            # Determine agent class based on insight reward scale
            if config.insight_reward_scale == 0.0 and 'Baseline' in config.experiment_name:
                agent_class = BaselineAgent
            else:
                agent_class = InsightSpikeAgent
            
            trial_results = []
            
            for trial in range(config.num_trials):
                result = self.run_single_trial(agent_class, config, trial)
                trial_results.append(result)
                
                if (trial + 1) % 5 == 0:
                    logger.info(f"Completed {trial + 1}/{config.num_trials} trials")
            
            results[config.experiment_name] = trial_results
            
            # Save intermediate results
            self.save_results({config.experiment_name: trial_results}, 
                            f"{config.experiment_name}_results.json")
        
        return results
    
    def save_results(self, results: Dict[str, List[TrialResult]], filename: str):
        """Save results to JSON file"""
        # Convert to serializable format
        serializable_results = {}
        for exp_name, trial_results in results.items():
            serializable_results[exp_name] = [asdict(trial) for trial in trial_results]
        
        output_file = self.output_dir / filename
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def statistical_analysis(self, results: Dict[str, List[TrialResult]]) -> Dict[str, Any]:
        """Perform statistical analysis on results"""
        
        analysis = {}
        
        # Extract metrics for each experiment
        for exp_name, trial_results in results.items():
            mean_rewards = [trial.mean_reward for trial in trial_results]
            final_success_rates = [trial.final_success_rate for trial in trial_results]
            times_to_success = [trial.time_to_first_success for trial in trial_results 
                              if trial.time_to_first_success is not None]
            total_insights = [len(trial.total_insights) if hasattr(trial.total_insights, '__len__') 
                            else trial.total_insights for trial in trial_results]
            
            analysis[exp_name] = {
                'mean_reward': {
                    'mean': np.mean(mean_rewards),
                    'std': np.std(mean_rewards),
                    'ci_lower': np.percentile(mean_rewards, 2.5),
                    'ci_upper': np.percentile(mean_rewards, 97.5)
                },
                'success_rate': {
                    'mean': np.mean(final_success_rates),
                    'std': np.std(final_success_rates),
                    'ci_lower': np.percentile(final_success_rates, 2.5),
                    'ci_upper': np.percentile(final_success_rates, 97.5)
                },
                'time_to_success': {
                    'mean': np.mean(times_to_success) if times_to_success else float('inf'),
                    'std': np.std(times_to_success) if times_to_success else 0,
                    'sample_size': len(times_to_success)
                },
                'insights': {
                    'mean': np.mean(total_insights),
                    'std': np.std(total_insights),
                    'total': sum(total_insights)
                }
            }
        
        # Pairwise comparisons
        exp_names = list(results.keys())
        if len(exp_names) >= 2:
            comparisons = {}
            
            for i in range(len(exp_names)):
                for j in range(i + 1, len(exp_names)):
                    exp1, exp2 = exp_names[i], exp_names[j]
                    
                    # T-test for mean rewards
                    rewards1 = [trial.mean_reward for trial in results[exp1]]
                    rewards2 = [trial.mean_reward for trial in results[exp2]]
                    
                    t_stat, p_value = stats.ttest_ind(rewards1, rewards2)
                    effect_size = (np.mean(rewards1) - np.mean(rewards2)) / np.sqrt(
                        (np.std(rewards1)**2 + np.std(rewards2)**2) / 2
                    )
                    
                    comparisons[f"{exp1}_vs_{exp2}"] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'effect_size': effect_size,
                        'improvement_percent': ((np.mean(rewards1) / np.mean(rewards2)) - 1) * 100
                    }
            
            analysis['comparisons'] = comparisons
        
        return analysis


# Export main classes
__all__ = [
    'ExperimentConfig',
    'TrialResult', 
    'BaselineAgent',
    'InsightSpikeAgent',
    'SimpleEnvironment',
    'ExperimentRunner'
]
