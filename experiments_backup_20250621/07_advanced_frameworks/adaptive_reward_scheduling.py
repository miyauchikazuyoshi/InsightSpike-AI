"""
Adaptive Reward Scheduling for InsightSpike-AI
=============================================

Advanced reward scheduling system that adapts to learning progress,
environmental complexity, and agent performance.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from abc import ABC, abstractmethod
from enum import Enum
import math

logger = logging.getLogger(__name__)


class LearningPhase(Enum):
    """Learning phases for adaptive scheduling"""
    EXPLORATION = "exploration"
    SKILL_ACQUISITION = "skill_acquisition"
    OPTIMIZATION = "optimization"
    MASTERY = "mastery"


@dataclass
class AdaptiveScheduleConfig:
    """Configuration for adaptive reward scheduling"""
    # Phase detection parameters
    exploration_threshold: float = 0.3
    skill_acquisition_threshold: float = 0.6
    optimization_threshold: float = 0.8
    mastery_threshold: float = 0.95
    
    # Performance window for phase detection
    performance_window: int = 20
    stability_threshold: float = 0.05
    
    # Reward scaling factors by phase
    exploration_factors: Dict[str, float] = field(default_factory=lambda: {
        'curiosity': 1.0, 'novelty': 1.0, 'insight': 0.5, 'exploitation': 0.2
    })
    skill_acquisition_factors: Dict[str, float] = field(default_factory=lambda: {
        'curiosity': 0.7, 'novelty': 0.8, 'insight': 1.0, 'exploitation': 0.4
    })
    optimization_factors: Dict[str, float] = field(default_factory=lambda: {
        'curiosity': 0.4, 'novelty': 0.5, 'insight': 1.2, 'exploitation': 0.8
    })
    mastery_factors: Dict[str, float] = field(default_factory=lambda: {
        'curiosity': 0.2, 'novelty': 0.3, 'insight': 0.8, 'exploitation': 1.0
    })
    
    # Dynamic adaptation parameters
    adaptation_rate: float = 0.1
    momentum: float = 0.9
    temperature_decay: float = 0.99
    
    # Curriculum learning
    curriculum_enabled: bool = True
    difficulty_ramp_episodes: int = 100
    max_difficulty_multiplier: float = 2.0


class PerformanceMonitor:
    """Monitor agent performance for adaptive scheduling"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.performance_history = deque(maxlen=window_size)
        self.reward_history = deque(maxlen=window_size)
        self.success_history = deque(maxlen=window_size)
        self.exploration_history = deque(maxlen=window_size)
        
    def update(self, performance: float, reward: float, success: bool, exploration_rate: float):
        """Update performance metrics"""
        self.performance_history.append(performance)
        self.reward_history.append(reward)
        self.success_history.append(1.0 if success else 0.0)
        self.exploration_history.append(exploration_rate)
    
    def get_current_performance(self) -> float:
        """Get current performance level (0-1)"""
        if not self.performance_history:
            return 0.0
        return np.mean(list(self.performance_history))
    
    def get_performance_stability(self) -> float:
        """Get performance stability (lower = more stable)"""
        if len(self.performance_history) < 10:
            return 1.0
        return np.std(list(self.performance_history))
    
    def get_learning_rate(self) -> float:
        """Estimate current learning rate"""
        if len(self.performance_history) < 10:
            return 0.0
        
        recent = list(self.performance_history)[-10:]
        older = list(self.performance_history)[-20:-10] if len(self.performance_history) >= 20 else recent
        
        return np.mean(recent) - np.mean(older)
    
    def get_exploration_efficiency(self) -> float:
        """Get exploration efficiency metric"""
        if len(self.exploration_history) < 5:
            return 0.5
        
        exploration_mean = np.mean(list(self.exploration_history))
        performance_mean = np.mean(list(self.performance_history))
        
        # Higher exploration with higher performance indicates efficient exploration
        return min(1.0, performance_mean / max(0.1, exploration_mean))


class PhaseDetector:
    """Detect current learning phase based on performance metrics"""
    
    def __init__(self, config: AdaptiveScheduleConfig):
        self.config = config
        self.current_phase = LearningPhase.EXPLORATION
        self.phase_history = []
        self.phase_transition_episodes = {}
        
    def detect_phase(self, monitor: PerformanceMonitor, episode: int) -> LearningPhase:
        """Detect current learning phase"""
        performance = monitor.get_current_performance()
        stability = monitor.get_performance_stability()
        learning_rate = monitor.get_learning_rate()
        
        # Phase detection logic
        if performance < self.config.exploration_threshold:
            phase = LearningPhase.EXPLORATION
        elif performance < self.config.skill_acquisition_threshold:
            phase = LearningPhase.SKILL_ACQUISITION
        elif performance < self.config.optimization_threshold:
            phase = LearningPhase.OPTIMIZATION
        else:
            # Check for mastery (high performance + stability)
            if stability < self.config.stability_threshold:
                phase = LearningPhase.MASTERY
            else:
                phase = LearningPhase.OPTIMIZATION
        
        # Track phase transitions
        if phase != self.current_phase:
            self.phase_transition_episodes[phase] = episode
            logger.info(f"Phase transition: {self.current_phase} -> {phase} at episode {episode}")
        
        self.current_phase = phase
        self.phase_history.append((episode, phase))
        
        return phase


class CurriculumManager:
    """Manage curriculum learning for adaptive difficulty"""
    
    def __init__(self, config: AdaptiveScheduleConfig):
        self.config = config
        self.current_difficulty = 1.0
        self.performance_targets = deque(maxlen=10)
        self.difficulty_history = []
        
    def update_difficulty(self, performance: float, episode: int) -> float:
        """Update curriculum difficulty based on performance"""
        if not self.config.curriculum_enabled:
            return 1.0
        
        self.performance_targets.append(performance)
        
        # Calculate target difficulty based on episode and performance
        episode_progress = min(1.0, episode / self.config.difficulty_ramp_episodes)
        base_difficulty = 1.0 + (self.config.max_difficulty_multiplier - 1.0) * episode_progress
        
        # Adjust based on recent performance
        if len(self.performance_targets) >= 5:
            recent_performance = np.mean(list(self.performance_targets))
            
            # If performing well, increase difficulty
            if recent_performance > 0.8:
                self.current_difficulty = min(
                    self.config.max_difficulty_multiplier,
                    self.current_difficulty * 1.05
                )
            # If struggling, decrease difficulty
            elif recent_performance < 0.4:
                self.current_difficulty = max(0.5, self.current_difficulty * 0.95)
        
        # Combine base and adaptive difficulty
        final_difficulty = 0.7 * base_difficulty + 0.3 * self.current_difficulty
        self.difficulty_history.append((episode, final_difficulty))
        
        return final_difficulty


class DynamicRewardScheduler:
    """Dynamic reward scheduler that adapts to learning progress"""
    
    def __init__(self, config: AdaptiveScheduleConfig):
        self.config = config
        self.phase_detector = PhaseDetector(config)
        self.performance_monitor = PerformanceMonitor()
        self.curriculum_manager = CurriculumManager(config)
        
        # Adaptive parameters
        self.reward_weights = {
            'curiosity': 1.0,
            'novelty': 1.0,
            'insight': 1.0,
            'exploitation': 1.0
        }
        self.weight_momentum = {key: 0.0 for key in self.reward_weights}
        self.temperature = 1.0
        
        # History tracking
        self.weight_history = []
        self.decision_history = []
        
    def update_performance(self, performance: float, reward: float, 
                         success: bool, exploration_rate: float, episode: int):
        """Update performance monitoring"""
        self.performance_monitor.update(performance, reward, success, exploration_rate)
        
        # Detect current phase
        current_phase = self.phase_detector.detect_phase(self.performance_monitor, episode)
        
        # Update curriculum difficulty
        difficulty = self.curriculum_manager.update_difficulty(performance, episode)
        
        # Adapt reward weights
        self._adapt_reward_weights(current_phase, performance, episode)
        
        return current_phase, difficulty
    
    def get_reward_weights(self, episode: int) -> Dict[str, float]:
        """Get current reward weights"""
        # Apply temperature scaling for exploration/exploitation balance
        scaled_weights = {}
        for key, weight in self.reward_weights.items():
            scaled_weights[key] = weight * self.temperature
        
        # Normalize weights
        total_weight = sum(scaled_weights.values())
        if total_weight > 0:
            for key in scaled_weights:
                scaled_weights[key] /= total_weight
        
        # Update temperature
        self.temperature *= self.config.temperature_decay
        
        return scaled_weights
    
    def _adapt_reward_weights(self, phase: LearningPhase, performance: float, episode: int):
        """Adapt reward weights based on current phase and performance"""
        
        # Get target weights for current phase
        if phase == LearningPhase.EXPLORATION:
            target_weights = self.config.exploration_factors
        elif phase == LearningPhase.SKILL_ACQUISITION:
            target_weights = self.config.skill_acquisition_factors
        elif phase == LearningPhase.OPTIMIZATION:
            target_weights = self.config.optimization_factors
        else:  # MASTERY
            target_weights = self.config.mastery_factors
        
        # Gradual adaptation with momentum
        for key in self.reward_weights:
            if key in target_weights:
                # Calculate target adjustment
                target_adjustment = target_weights[key] - self.reward_weights[key]
                
                # Apply momentum
                self.weight_momentum[key] = (
                    self.config.momentum * self.weight_momentum[key] +
                    self.config.adaptation_rate * target_adjustment
                )
                
                # Update weight
                self.reward_weights[key] += self.weight_momentum[key]
                
                # Clamp to reasonable range
                self.reward_weights[key] = max(0.0, min(2.0, self.reward_weights[key]))
        
        # Track history
        self.weight_history.append({
            'episode': episode,
            'phase': phase.value,
            'weights': self.reward_weights.copy(),
            'performance': performance
        })
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive adaptation statistics"""
        return {
            'current_phase': self.phase_detector.current_phase.value,
            'current_weights': self.reward_weights.copy(),
            'current_temperature': self.temperature,
            'phase_transitions': self.phase_detector.phase_transition_episodes,
            'current_difficulty': self.curriculum_manager.current_difficulty,
            'performance_stats': {
                'current': self.performance_monitor.get_current_performance(),
                'stability': self.performance_monitor.get_performance_stability(),
                'learning_rate': self.performance_monitor.get_learning_rate(),
                'exploration_efficiency': self.performance_monitor.get_exploration_efficiency()
            },
            'weight_history': self.weight_history,
            'difficulty_history': self.curriculum_manager.difficulty_history
        }


class AdaptiveInsightSpikeAgent:
    """InsightSpike agent with adaptive reward scheduling"""
    
    def __init__(self, state_size: int, action_size: int,
                 schedule_config: Optional[AdaptiveScheduleConfig] = None):
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Q-learning parameters
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.3
        
        # Adaptive scheduling
        if schedule_config is None:
            schedule_config = AdaptiveScheduleConfig()
        
        self.scheduler = DynamicRewardScheduler(schedule_config)
        
        # Episode tracking
        self.current_episode = 0
        self.episode_rewards = []
        self.episode_successes = []
        
    def update_episode(self, episode_reward: float, episode_success: bool):
        """Update episode-level statistics"""
        self.episode_rewards.append(episode_reward)
        self.episode_successes.append(episode_success)
        
        # Calculate performance metrics
        performance = np.mean(self.episode_successes[-20:]) if self.episode_successes else 0.0
        exploration_rate = self.epsilon
        
        # Update scheduler
        phase, difficulty = self.scheduler.update_performance(
            performance, episode_reward, episode_success, exploration_rate, self.current_episode
        )
        
        self.current_episode += 1
        
        return phase, difficulty
    
    def choose_action(self, state: int) -> int:
        """Choose action with adaptive exploration"""
        # Get current reward weights for exploration adjustment
        weights = self.scheduler.get_reward_weights(self.current_episode)
        
        # Adjust epsilon based on curiosity weight
        adapted_epsilon = self.epsilon * weights.get('curiosity', 1.0)
        
        if np.random.random() < adapted_epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.q_table[state])
    
    def update_q_table(self, state: int, action: int, reward: float, next_state: int,
                      insight_detected: bool = False, insight_quality: float = 1.0):
        """Update Q-table with adaptive reward scaling"""
        
        # Get current reward weights
        weights = self.scheduler.get_reward_weights(self.current_episode)
        
        # Calculate adaptive reward
        base_reward = reward
        
        # Add insight bonus if detected
        if insight_detected:
            insight_bonus = weights.get('insight', 1.0) * insight_quality
            base_reward += insight_bonus
        
        # Apply exploitation/exploration balance
        exploration_bonus = weights.get('curiosity', 1.0) * 0.1 * np.random.random()
        adapted_reward = base_reward + exploration_bonus
        
        # Standard Q-learning update
        current_q = self.q_table[state, action]
        next_max_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (
            adapted_reward + self.discount_factor * next_max_q - current_q
        )
        self.q_table[state, action] = new_q
        
        return adapted_reward
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        base_stats = {
            'episode': self.current_episode,
            'total_episodes': len(self.episode_rewards),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'success_rate': np.mean(self.episode_successes) if self.episode_successes else 0.0,
            'current_epsilon': self.epsilon
        }
        
        # Add adaptive scheduling statistics
        adaptation_stats = self.scheduler.get_adaptation_statistics()
        
        return {**base_stats, 'adaptation': adaptation_stats}


def run_adaptive_scheduling_experiment():
    """Run experiment to demonstrate adaptive reward scheduling"""
    
    # Create adaptive configuration
    config = AdaptiveScheduleConfig(
        exploration_threshold=0.2,
        skill_acquisition_threshold=0.5,
        optimization_threshold=0.8,
        curriculum_enabled=True,
        adaptation_rate=0.05
    )
    
    # Create adaptive agent
    agent = AdaptiveInsightSpikeAgent(
        state_size=100,
        action_size=4,
        schedule_config=config
    )
    
    # Run training episodes
    results = []
    for episode in range(200):
        episode_reward = 0
        episode_success = False
        state = np.random.randint(0, 100)
        
        # Simulate episode
        for step in range(50):
            action = agent.choose_action(state)
            next_state = np.random.randint(0, 100)
            
            # Simulate reward (gradually improving)
            base_reward = -0.1 + 0.001 * episode + np.random.normal(0, 0.1)
            
            # Simulate insight detection (increasing with learning)
            insight_prob = min(0.3, 0.01 * episode)
            insight_detected = np.random.random() < insight_prob
            insight_quality = np.random.random() if insight_detected else 0.0
            
            # Update agent
            adapted_reward = agent.update_q_table(
                state, action, base_reward, next_state, 
                insight_detected, insight_quality
            )
            
            episode_reward += adapted_reward
            state = next_state
            
            # Check for episode success (improving with learning)
            if np.random.random() < min(0.8, 0.005 * episode):
                episode_success = True
                break
        
        # Update episode statistics
        phase, difficulty = agent.update_episode(episode_reward, episode_success)
        
        # Decay epsilon
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        
        # Store results
        if episode % 10 == 0:
            stats = agent.get_training_statistics()
            results.append({
                'episode': episode,
                'phase': phase.value,
                'difficulty': difficulty,
                'performance': stats['success_rate'],
                'avg_reward': stats['average_reward'],
                'weights': stats['adaptation']['current_weights'].copy()
            })
    
    return results, agent.get_training_statistics()


def analyze_adaptive_results(results: List[Dict], final_stats: Dict):
    """Analyze results from adaptive scheduling experiment"""
    
    print("Adaptive Reward Scheduling Experiment Results")
    print("=" * 50)
    
    # Episode progression
    episodes = [r['episode'] for r in results]
    phases = [r['phase'] for r in results]
    performance = [r['performance'] for r in results]
    difficulties = [r['difficulty'] for r in results]
    
    print(f"Total Episodes: {max(episodes) + 1}")
    print(f"Final Performance: {performance[-1]:.3f}")
    print(f"Final Phase: {phases[-1]}")
    print(f"Final Difficulty: {difficulties[-1]:.3f}")
    
    # Phase transitions
    print("\nPhase Transitions:")
    current_phase = None
    for episode, phase in zip(episodes, phases):
        if phase != current_phase:
            print(f"  Episode {episode}: -> {phase}")
            current_phase = phase
    
    # Weight evolution
    print("\nFinal Reward Weights:")
    final_weights = final_stats['adaptation']['current_weights']
    for weight_type, value in final_weights.items():
        print(f"  {weight_type}: {value:.3f}")
    
    # Performance statistics
    perf_stats = final_stats['adaptation']['performance_stats']
    print(f"\nPerformance Statistics:")
    print(f"  Current Performance: {perf_stats['current']:.3f}")
    print(f"  Stability: {perf_stats['stability']:.3f}")
    print(f"  Learning Rate: {perf_stats['learning_rate']:.3f}")
    print(f"  Exploration Efficiency: {perf_stats['exploration_efficiency']:.3f}")


if __name__ == "__main__":
    # Run adaptive scheduling experiment
    print("Running adaptive reward scheduling experiment...")
    results, final_stats = run_adaptive_scheduling_experiment()
    
    # Analyze results
    analyze_adaptive_results(results, final_stats)
    
    print("\nAdaptive reward scheduling experiment completed successfully!")
