"""
InsightSpike-AI Bias-Corrected Evaluation Framework
==================================================

This module implements experimental evaluation with improved methodological rigor:
- Multiple independent baseline implementations
- Diverse evaluation environments 
- Proper statistical controls
- Independent validation protocols
"""

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import logging
from scipy import stats
import pandas as pd
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor
import warnings

# Suppress statistical warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

@dataclass
class ExperimentConfig:
    """Configuration for bias-corrected experiments"""
    name: str
    description: str
    episodes_per_run: int = 200  # Increased from 50
    independent_runs: int = 30   # Increased from 5
    random_seeds: List[int] = None
    environment_variants: List[str] = None
    
    def __post_init__(self):
        if self.random_seeds is None:
            # Use predetermined seeds for reproducibility
            self.random_seeds = list(range(42, 42 + self.independent_runs))
        if self.environment_variants is None:
            self.environment_variants = ['simple', 'noisy', 'dynamic', 'complex']

class IndependentBaselines:
    """Multiple independent baseline implementations to avoid implementation bias"""
    
    @staticmethod
    def vanilla_q_learning(state_size: int, action_size: int, **kwargs):
        """Standard Q-learning implementation"""
        return VanillaQLearning(state_size, action_size, **kwargs)
    
    @staticmethod
    def epsilon_greedy_ql(state_size: int, action_size: int, **kwargs):
        """Epsilon-greedy Q-learning with different exploration strategy"""
        return EpsilonGreedyQLearning(state_size, action_size, **kwargs)
    
    @staticmethod
    def ucb_exploration(state_size: int, action_size: int, **kwargs):
        """Upper Confidence Bound exploration"""
        return UCBQLearning(state_size, action_size, **kwargs)
    
    @staticmethod
    def random_baseline(state_size: int, action_size: int, **kwargs):
        """Random action baseline for lower bound"""
        return RandomAgent(action_size)

class VanillaQLearning:
    """Standard Q-learning implementation without bias"""
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.1, epsilon: float = 0.1, 
                 gamma: float = 0.95, epsilon_decay: float = 0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((state_size, action_size))
        self.name = "Vanilla Q-Learning"
    
    def get_action(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        current_q = self.q_table[state, action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        
        self.q_table[state, action] = current_q + self.lr * (target_q - current_q)
        
        if done:
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

class EpsilonGreedyQLearning:
    """Alternative Q-learning with different epsilon strategy"""
    
    def __init__(self, state_size: int, action_size: int,
                 learning_rate: float = 0.15, epsilon: float = 0.3,
                 gamma: float = 0.9, epsilon_min: float = 0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.q_table = np.zeros((state_size, action_size))
        self.name = "Epsilon-Greedy Q-Learning"
    
    def get_action(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        current_q = self.q_table[state, action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        
        self.q_table[state, action] = current_q + self.lr * (target_q - current_q)
        
        # Linear epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (0.3 - self.epsilon_min) / 1000

class UCBQLearning:
    """Q-learning with Upper Confidence Bound exploration"""
    
    def __init__(self, state_size: int, action_size: int,
                 learning_rate: float = 0.1, c: float = 2.0, gamma: float = 0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.c = c
        self.gamma = gamma
        self.q_table = np.zeros((state_size, action_size))
        self.action_counts = np.ones((state_size, action_size))  # Initialize to 1 to avoid division by zero
        self.total_counts = np.ones(state_size)
        self.name = "UCB Q-Learning"
    
    def get_action(self, state: int) -> int:
        # UCB action selection
        ucb_values = self.q_table[state] + self.c * np.sqrt(
            np.log(self.total_counts[state]) / self.action_counts[state]
        )
        return np.argmax(ucb_values)
    
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        current_q = self.q_table[state, action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        
        self.q_table[state, action] = current_q + self.lr * (target_q - current_q)
        self.action_counts[state, action] += 1
        self.total_counts[state] += 1

class RandomAgent:
    """Random baseline for comparison"""
    
    def __init__(self, action_size: int):
        self.action_size = action_size
        self.name = "Random Agent"
    
    def get_action(self, state: int) -> int:
        return random.randint(0, self.action_size - 1)
    
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        pass  # Random agent doesn't learn

class SimplifiedInsightSpike:
    """Simplified InsightSpike-AI implementation for fair comparison"""
    
    def __init__(self, state_size: int, action_size: int,
                 learning_rate: float = 0.1, epsilon: float = 0.1,
                 gamma: float = 0.95, insight_threshold: float = 0.5):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.insight_threshold = insight_threshold
        self.q_table = np.zeros((state_size, action_size))
        self.insight_memory = []
        self.performance_history = []
        self.name = "InsightSpike-AI (Simplified)"
    
    def get_action(self, state: int) -> int:
        # Check for insight-based action selection
        if len(self.performance_history) > 10:
            recent_performance = np.mean(self.performance_history[-10:])
            if len(self.performance_history) > 20:
                older_performance = np.mean(self.performance_history[-20:-10])
                if recent_performance - older_performance > self.insight_threshold:
                    # Insight detected - exploit current knowledge
                    return np.argmax(self.q_table[state])
        
        # Standard epsilon-greedy
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        current_q = self.q_table[state, action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        
        self.q_table[state, action] = current_q + self.lr * (target_q - current_q)
        
        # Track performance for insight detection
        self.performance_history.append(reward)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        if done:
            self.epsilon = max(0.01, self.epsilon * 0.995)

class MazeEnvironment:
    """Simplified maze environment with multiple variants"""
    
    def __init__(self, variant: str = 'simple', seed: int = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.variant = variant
        self.size = 10 if variant in ['simple', 'noisy'] else 15
        self.noise_level = 0.1 if variant == 'noisy' else 0.0
        self.dynamic = variant == 'dynamic'
        self.complex = variant == 'complex'
        
        self.state_size = self.size * self.size
        self.action_size = 4  # up, down, left, right
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.agent_pos = (0, 0)
        self.goal_pos = (self.size - 1, self.size - 1)
        self.steps = 0
        self.max_steps = self.size * self.size * 2
        
        # Create obstacles
        self.obstacles = set()
        if self.complex:
            # More obstacles for complex variant
            num_obstacles = int(self.size * self.size * 0.2)
        else:
            num_obstacles = int(self.size * self.size * 0.1)
        
        for _ in range(num_obstacles):
            obs = (np.random.randint(self.size), np.random.randint(self.size))
            if obs != self.agent_pos and obs != self.goal_pos:
                self.obstacles.add(obs)
        
        return self._get_state()
    
    def _get_state(self):
        """Convert position to state index"""
        return self.agent_pos[0] * self.size + self.agent_pos[1]
    
    def step(self, action: int):
        """Take action and return next state, reward, done"""
        self.steps += 1
        
        # Add noise to action
        if self.noise_level > 0 and random.random() < self.noise_level:
            action = random.randint(0, 3)
        
        # Apply action
        new_pos = list(self.agent_pos)
        if action == 0:    # up
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 1:  # down
            new_pos[0] = min(self.size - 1, new_pos[0] + 1)
        elif action == 2:  # left
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 3:  # right
            new_pos[1] = min(self.size - 1, new_pos[1] + 1)
        
        new_pos = tuple(new_pos)
        
        # Check for obstacles
        if new_pos not in self.obstacles:
            self.agent_pos = new_pos
        
        # Dynamic environment changes
        if self.dynamic and self.steps % 50 == 0:
            # Randomly move some obstacles
            obstacles_list = list(self.obstacles)
            if obstacles_list:
                old_obs = random.choice(obstacles_list)
                self.obstacles.remove(old_obs)
                new_obs = (np.random.randint(self.size), np.random.randint(self.size))
                if new_obs != self.agent_pos and new_obs != self.goal_pos:
                    self.obstacles.add(new_obs)
        
        # Calculate reward
        if self.agent_pos == self.goal_pos:
            reward = 100.0
            done = True
        elif self.steps >= self.max_steps:
            reward = -10.0
            done = True
        else:
            # Distance-based reward
            distance = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
            reward = -0.1 - distance * 0.01
            done = False
        
        return self._get_state(), reward, done

class RigorousStatisticalAnalysis:
    """Statistical analysis with proper controls and multiple testing correction"""
    
    @staticmethod
    def compare_algorithms(results_dict: Dict[str, List[float]], 
                          alpha: float = 0.05) -> Dict[str, Any]:
        """Compare multiple algorithms with statistical rigor"""
        
        algorithms = list(results_dict.keys())
        n_algorithms = len(algorithms)
        
        # Bonferroni correction for multiple comparisons
        corrected_alpha = alpha / (n_algorithms * (n_algorithms - 1) / 2)
        
        analysis = {
            'descriptive': {},
            'normality_tests': {},
            'pairwise_comparisons': {},
            'effect_sizes': {},
            'corrected_alpha': corrected_alpha
        }
        
        # Descriptive statistics
        for alg in algorithms:
            data = np.array(results_dict[alg])
            analysis['descriptive'][alg] = {
                'mean': np.mean(data),
                'median': np.median(data),
                'std': np.std(data, ddof=1),
                'min': np.min(data),
                'max': np.max(data),
                'q25': np.percentile(data, 25),
                'q75': np.percentile(data, 75),
                'n': len(data)
            }
        
        # Normality tests
        for alg in algorithms:
            data = np.array(results_dict[alg])
            if len(data) >= 3:
                stat, p = stats.shapiro(data)
                analysis['normality_tests'][alg] = {'statistic': stat, 'p_value': p}
        
        # Pairwise comparisons
        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms):
                if i < j:
                    data1 = np.array(results_dict[alg1])
                    data2 = np.array(results_dict[alg2])
                    
                    # Choose appropriate test based on normality
                    normal1 = analysis['normality_tests'].get(alg1, {}).get('p_value', 0) > 0.05
                    normal2 = analysis['normality_tests'].get(alg2, {}).get('p_value', 0) > 0.05
                    
                    if normal1 and normal2:
                        # Welch's t-test (unequal variances)
                        stat, p = stats.ttest_ind(data1, data2, equal_var=False)
                        test_name = "Welch's t-test"
                    else:
                        # Mann-Whitney U test
                        stat, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                        test_name = "Mann-Whitney U"
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                        (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                       (len(data1) + len(data2) - 2))
                    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
                    
                    comparison_key = f"{alg1} vs {alg2}"
                    analysis['pairwise_comparisons'][comparison_key] = {
                        'test': test_name,
                        'statistic': stat,
                        'p_value': p,
                        'significant': p < corrected_alpha,
                        'corrected_alpha': corrected_alpha
                    }
                    
                    analysis['effect_sizes'][comparison_key] = {
                        'cohens_d': cohens_d,
                        'interpretation': RigorousStatisticalAnalysis._interpret_effect_size(abs(cohens_d))
                    }
        
        return analysis
    
    @staticmethod
    def _interpret_effect_size(d: float) -> str:
        """Interpret Cohen's d effect size"""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

class BiasCorrectdEvaluationFramework:
    """Main framework for bias-corrected evaluation"""
    
    def __init__(self, output_dir: str = "experiments/outputs/bias_corrected"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_comprehensive_evaluation(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run comprehensive evaluation across multiple baselines and environments"""
        
        self.logger.info(f"Starting evaluation: {config.name}")
        self.logger.info(f"Episodes per run: {config.episodes_per_run}")
        self.logger.info(f"Independent runs: {config.independent_runs}")
        
        # Initialize baseline algorithms
        baselines = {
            'Random': IndependentBaselines.random_baseline,
            'Vanilla Q-Learning': IndependentBaselines.vanilla_q_learning,
            'Epsilon-Greedy QL': IndependentBaselines.epsilon_greedy_ql,
            'UCB Q-Learning': IndependentBaselines.ucb_exploration
        }
        
        # Add InsightSpike for comparison
        algorithms = dict(baselines)
        algorithms['InsightSpike-AI'] = lambda state_size, action_size, **kwargs: SimplifiedInsightSpike(state_size, action_size, **kwargs)
        
        results = {}
        detailed_results = {}
        
        # Run experiments for each environment variant
        for env_variant in config.environment_variants:
            self.logger.info(f"Testing environment variant: {env_variant}")
            
            variant_results = {}
            variant_detailed = {}
            
            for alg_name, alg_factory in algorithms.items():
                self.logger.info(f"  Running algorithm: {alg_name}")
                
                alg_results = []
                alg_detailed = []
                
                # Run multiple independent trials
                for run_idx in range(config.independent_runs):
                    seed = config.random_seeds[run_idx]
                    
                    # Set random seeds for reproducibility
                    np.random.seed(seed)
                    random.seed(seed)
                    torch.manual_seed(seed)
                    
                    env = MazeEnvironment(variant=env_variant, seed=seed)
                    agent = alg_factory(env.state_size, env.action_size)
                    
                    # Run single trial
                    trial_result = self._run_single_trial(env, agent, config.episodes_per_run)
                    alg_results.append(trial_result['final_success_rate'])
                    alg_detailed.append(trial_result)
                
                variant_results[alg_name] = alg_results
                variant_detailed[alg_name] = alg_detailed
            
            results[env_variant] = variant_results
            detailed_results[env_variant] = variant_detailed
        
        # Perform statistical analysis
        statistical_analysis = {}
        for env_variant in config.environment_variants:
            statistical_analysis[env_variant] = RigorousStatisticalAnalysis.compare_algorithms(
                results[env_variant]
            )
        
        # Save results
        self._save_results({
            'config': config.__dict__,
            'results': results,
            'detailed_results': detailed_results,
            'statistical_analysis': statistical_analysis
        })
        
        return {
            'results': results,
            'statistical_analysis': statistical_analysis,
            'detailed_results': detailed_results
        }
    
    def _run_single_trial(self, env: MazeEnvironment, agent, episodes: int) -> Dict[str, Any]:
        """Run a single trial with an agent in an environment"""
        
        episode_rewards = []
        episode_lengths = []
        success_episodes = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if steps > 1000:  # Prevent infinite loops
                    break
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            success_episodes.append(1 if env.agent_pos == env.goal_pos else 0)
        
        # Calculate metrics
        recent_episodes = min(50, episodes // 4)  # Last quarter or 50 episodes
        final_success_rate = np.mean(success_episodes[-recent_episodes:]) if recent_episodes > 0 else 0
        final_reward = np.mean(episode_rewards[-recent_episodes:]) if recent_episodes > 0 else 0
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'success_episodes': success_episodes,
            'final_success_rate': final_success_rate,
            'final_reward': final_reward,
            'total_episodes': episodes
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """Save experimental results"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_path = self.output_dir / f"evaluation_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._convert_for_json(results)
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Results saved to {json_path}")
    
    def _convert_for_json(self, obj):
        """Convert numpy arrays and other non-JSON types for serialization"""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        else:
            return obj

def run_bias_corrected_evaluation():
    """Run the bias-corrected evaluation"""
    
    # Configuration with more conservative parameters
    config = ExperimentConfig(
        name="Bias-Corrected InsightSpike Evaluation",
        description="Evaluation with improved methodological rigor and multiple baselines",
        episodes_per_run=200,      # Increased sample size
        independent_runs=30,       # More independent trials
        environment_variants=['simple', 'noisy', 'dynamic', 'complex']
    )
    
    # Run evaluation
    framework = BiasCorrectdEvaluationFramework()
    results = framework.run_comprehensive_evaluation(config)
    
    # Print summary
    print("\n" + "="*80)
    print("BIAS-CORRECTED EVALUATION SUMMARY")
    print("="*80)
    
    for env_variant in config.environment_variants:
        print(f"\nEnvironment: {env_variant.upper()}")
        print("-" * 40)
        
        # Print means and confidence intervals
        for alg_name, alg_results in results['results'][env_variant].items():
            mean_val = np.mean(alg_results)
            std_val = np.std(alg_results, ddof=1)
            ci_95 = 1.96 * std_val / np.sqrt(len(alg_results))
            print(f"{alg_name:20}: {mean_val:.3f} ± {ci_95:.3f} (95% CI)")
        
        # Print significant comparisons
        stats_analysis = results['statistical_analysis'][env_variant]
        print(f"\nStatistical Comparisons (α = {stats_analysis['corrected_alpha']:.4f}):")
        for comparison, result in stats_analysis['pairwise_comparisons'].items():
            if result['significant']:
                effect = stats_analysis['effect_sizes'][comparison]
                print(f"  {comparison}: p = {result['p_value']:.4f} *, "
                      f"d = {effect['cohens_d']:.3f} ({effect['interpretation']})")
    
    return results

if __name__ == "__main__":
    results = run_bias_corrected_evaluation()
