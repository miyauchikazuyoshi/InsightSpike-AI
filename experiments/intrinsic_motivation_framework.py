"""
Intrinsic Motivation Framework for InsightSpike-AI
================================================

Advanced framework for integrating intrinsic motivation and curiosity-driven learning
with InsightSpike's insight detection mechanism.
"""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class IntrinsicRewardConfig:
    """Configuration for intrinsic reward mechanisms"""
    # Curiosity-based rewards
    curiosity_weight: float = 0.1
    novelty_threshold: float = 0.5
    exploration_bonus_decay: float = 0.99
    
    # Information gain rewards
    information_gain_weight: float = 0.05
    entropy_threshold: float = 1.0
    
    # Insight-based rewards
    insight_discovery_reward: float = 1.0
    insight_quality_multiplier: float = 2.0
    insight_persistence_bonus: float = 0.5
    
    # Adaptive scheduling
    adaptive_scheduling: bool = True
    reward_schedule_phases: List[float] = field(default_factory=lambda: [0.8, 0.5, 0.2])
    phase_transition_episodes: List[int] = field(default_factory=lambda: [30, 60, 90])
    
    # Meta-learning components
    meta_learning_enabled: bool = True
    reward_prediction_weight: float = 0.1
    surprise_threshold: float = 0.3


class IntrinsicMotivationModule(ABC):
    """Abstract base class for intrinsic motivation modules"""
    
    @abstractmethod
    def compute_reward(self, state: Any, action: Any, next_state: Any, 
                      context: Dict[str, Any]) -> float:
        """Compute intrinsic reward for the given transition"""
        pass
    
    @abstractmethod
    def update(self, experience: Dict[str, Any]) -> None:
        """Update internal parameters based on experience"""
        pass


class CuriosityModule(IntrinsicMotivationModule):
    """Curiosity-driven intrinsic motivation based on prediction error"""
    
    def __init__(self, config: IntrinsicRewardConfig):
        self.config = config
        self.state_predictions = {}
        self.prediction_errors = deque(maxlen=1000)
        self.visitation_counts = defaultdict(int)
        
    def compute_reward(self, state: Any, action: Any, next_state: Any, 
                      context: Dict[str, Any]) -> float:
        """Compute curiosity reward based on prediction error and novelty"""
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        # Novelty bonus (inverse visitation count)
        novelty_bonus = 1.0 / (1.0 + self.visitation_counts[next_state_key])
        
        # Prediction error (simplified as state transition surprise)
        prediction_error = self._compute_prediction_error(state, action, next_state)
        
        # Combine curiosity components
        curiosity_reward = (
            self.config.curiosity_weight * prediction_error +
            novelty_bonus * self.config.exploration_bonus_decay ** context.get('episode', 0)
        )
        
        return curiosity_reward
    
    def update(self, experience: Dict[str, Any]) -> None:
        """Update prediction model and visitation counts"""
        state = experience['state']
        next_state = experience['next_state']
        
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        self.visitation_counts[next_state_key] += 1
        
        # Update prediction errors
        error = self._compute_prediction_error(state, experience['action'], next_state)
        self.prediction_errors.append(error)
    
    def _state_to_key(self, state: Any) -> str:
        """Convert state to hashable key"""
        if isinstance(state, (tuple, list)):
            return str(tuple(state))
        return str(state)
    
    def _compute_prediction_error(self, state: Any, action: Any, next_state: Any) -> float:
        """Compute prediction error for state transition"""
        # Simplified prediction error based on state difference
        if isinstance(state, (tuple, list)) and isinstance(next_state, (tuple, list)):
            return np.linalg.norm(np.array(next_state) - np.array(state))
        return random.random()  # Fallback for non-numeric states


class InformationGainModule(IntrinsicMotivationModule):
    """Information gain based intrinsic motivation"""
    
    def __init__(self, config: IntrinsicRewardConfig):
        self.config = config
        self.state_distribution = defaultdict(int)
        self.total_observations = 0
        
    def compute_reward(self, state: Any, action: Any, next_state: Any, 
                      context: Dict[str, Any]) -> float:
        """Compute information gain reward"""
        next_state_key = self._state_to_key(next_state)
        
        # Current probability of the state
        current_prob = self.state_distribution[next_state_key] / max(1, self.total_observations)
        
        # Information gain (negative log probability)
        information_gain = -np.log(max(current_prob, 1e-10))
        
        # Apply threshold and weight
        if information_gain > self.config.entropy_threshold:
            return self.config.information_gain_weight * information_gain
        
        return 0.0
    
    def update(self, experience: Dict[str, Any]) -> None:
        """Update state distribution"""
        next_state_key = self._state_to_key(experience['next_state'])
        self.state_distribution[next_state_key] += 1
        self.total_observations += 1
    
    def _state_to_key(self, state: Any) -> str:
        """Convert state to hashable key"""
        if isinstance(state, (tuple, list)):
            return str(tuple(state))
        return str(state)


class InsightRewardModule(IntrinsicMotivationModule):
    """Insight-based intrinsic motivation"""
    
    def __init__(self, config: IntrinsicRewardConfig):
        self.config = config
        self.insight_history = []
        self.insight_quality_tracker = defaultdict(list)
        
    def compute_reward(self, state: Any, action: Any, next_state: Any, 
                      context: Dict[str, Any]) -> float:
        """Compute insight-based reward"""
        insight_reward = 0.0
        
        # Check if insight was detected
        if context.get('insight_detected', False):
            base_reward = self.config.insight_discovery_reward
            
            # Quality multiplier
            insight_quality = context.get('insight_quality', 1.0)
            quality_bonus = insight_quality * self.config.insight_quality_multiplier
            
            # Persistence bonus (reward insights that lead to sustained improvement)
            persistence_bonus = self._compute_persistence_bonus(context)
            
            insight_reward = base_reward + quality_bonus + persistence_bonus
            
            # Track insight
            self.insight_history.append({
                'episode': context.get('episode', 0),
                'step': context.get('step', 0),
                'quality': insight_quality,
                'reward': insight_reward
            })
        
        return insight_reward
    
    def update(self, experience: Dict[str, Any]) -> None:
        """Update insight tracking"""
        if experience.get('insight_detected', False):
            insight_type = experience.get('insight_type', 'general')
            quality = experience.get('insight_quality', 1.0)
            self.insight_quality_tracker[insight_type].append(quality)
    
    def _compute_persistence_bonus(self, context: Dict[str, Any]) -> float:
        """Compute bonus for insights that lead to sustained improvement"""
        recent_performance = context.get('recent_performance', [])
        if len(recent_performance) < 2:
            return 0.0
        
        # Check if performance has improved consistently
        improvement_trend = np.mean(np.diff(recent_performance))
        if improvement_trend > 0:
            return self.config.insight_persistence_bonus * improvement_trend
        
        return 0.0


class AdaptiveRewardScheduler:
    """Adaptive reward scheduling for intrinsic motivation"""
    
    def __init__(self, config: IntrinsicRewardConfig):
        self.config = config
        self.current_phase = 0
        self.phase_weights = {}
        
    def get_current_weights(self, episode: int) -> Dict[str, float]:
        """Get current reward weights based on episode"""
        if not self.config.adaptive_scheduling:
            return {'curiosity': 1.0, 'information_gain': 1.0, 'insight': 1.0}
        
        # Determine current phase
        phase = 0
        for i, transition_episode in enumerate(self.config.phase_transition_episodes):
            if episode >= transition_episode:
                phase = i + 1
        
        # Get phase-specific weights
        if phase < len(self.config.reward_schedule_phases):
            curiosity_weight = self.config.reward_schedule_phases[phase]
            information_weight = max(0.1, 1.0 - curiosity_weight)
            insight_weight = 1.0  # Insight rewards remain constant
        else:
            # Final phase: minimal intrinsic, focus on exploitation
            curiosity_weight = 0.1
            information_weight = 0.1
            insight_weight = 1.0
        
        return {
            'curiosity': curiosity_weight,
            'information_gain': information_weight,
            'insight': insight_weight
        }


class MetaLearningRewardPredictor:
    """Meta-learning component for reward prediction and adaptation"""
    
    def __init__(self, config: IntrinsicRewardConfig):
        self.config = config
        self.reward_history = deque(maxlen=1000)
        self.prediction_errors = deque(maxlen=100)
        
    def predict_reward(self, state: Any, action: Any, context: Dict[str, Any]) -> float:
        """Predict expected reward for state-action pair"""
        # Simplified prediction based on historical averages
        if len(self.reward_history) < 10:
            return 0.0
        
        recent_rewards = list(self.reward_history)[-50:]
        return np.mean(recent_rewards)
    
    def update_prediction(self, predicted_reward: float, actual_reward: float) -> None:
        """Update prediction model"""
        error = abs(predicted_reward - actual_reward)
        self.prediction_errors.append(error)
        self.reward_history.append(actual_reward)
    
    def compute_surprise_bonus(self, predicted_reward: float, actual_reward: float) -> float:
        """Compute surprise bonus for unexpected rewards"""
        surprise = abs(predicted_reward - actual_reward)
        if surprise > self.config.surprise_threshold:
            return self.config.reward_prediction_weight * surprise
        return 0.0


class IntrinsicMotivationFramework:
    """Comprehensive framework for intrinsic motivation in InsightSpike-AI"""
    
    def __init__(self, config: IntrinsicRewardConfig):
        self.config = config
        
        # Initialize motivation modules
        self.curiosity_module = CuriosityModule(config)
        self.information_gain_module = InformationGainModule(config)
        self.insight_module = InsightRewardModule(config)
        
        # Initialize adaptive components
        self.scheduler = AdaptiveRewardScheduler(config)
        
        # Initialize meta-learning components
        if config.meta_learning_enabled:
            self.meta_learner = MetaLearningRewardPredictor(config)
        else:
            self.meta_learner = None
        
        # Tracking
        self.total_intrinsic_reward = 0.0
        self.reward_breakdown = defaultdict(float)
        
    def compute_intrinsic_reward(self, state: Any, action: Any, next_state: Any,
                                context: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Compute total intrinsic reward and breakdown"""
        episode = context.get('episode', 0)
        
        # Get adaptive weights
        weights = self.scheduler.get_current_weights(episode)
        
        # Compute component rewards
        curiosity_reward = self.curiosity_module.compute_reward(state, action, next_state, context)
        info_gain_reward = self.information_gain_module.compute_reward(state, action, next_state, context)
        insight_reward = self.insight_module.compute_reward(state, action, next_state, context)
        
        # Apply adaptive weights
        weighted_curiosity = weights['curiosity'] * curiosity_reward
        weighted_info_gain = weights['information_gain'] * info_gain_reward
        weighted_insight = weights['insight'] * insight_reward
        
        # Meta-learning surprise bonus
        surprise_bonus = 0.0
        if self.meta_learner:
            predicted_reward = self.meta_learner.predict_reward(state, action, context)
            total_reward = weighted_curiosity + weighted_info_gain + weighted_insight
            surprise_bonus = self.meta_learner.compute_surprise_bonus(predicted_reward, total_reward)
            
            # Update meta-learner
            self.meta_learner.update_prediction(predicted_reward, total_reward)
        
        # Total intrinsic reward
        total_intrinsic = weighted_curiosity + weighted_info_gain + weighted_insight + surprise_bonus
        
        # Track rewards
        self.total_intrinsic_reward += total_intrinsic
        reward_breakdown = {
            'curiosity': weighted_curiosity,
            'information_gain': weighted_info_gain,
            'insight': weighted_insight,
            'surprise': surprise_bonus,
            'total': total_intrinsic
        }
        
        for key, value in reward_breakdown.items():
            self.reward_breakdown[key] += value
        
        return total_intrinsic, reward_breakdown
    
    def update_modules(self, experience: Dict[str, Any]) -> None:
        """Update all motivation modules with experience"""
        self.curiosity_module.update(experience)
        self.information_gain_module.update(experience)
        self.insight_module.update(experience)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get framework statistics"""
        stats = {
            'total_intrinsic_reward': self.total_intrinsic_reward,
            'reward_breakdown': dict(self.reward_breakdown),
            'num_insights': len(self.insight_module.insight_history),
            'insight_history': self.insight_module.insight_history
        }
        
        if self.meta_learner:
            stats['meta_learning'] = {
                'avg_prediction_error': np.mean(list(self.meta_learner.prediction_errors)) if self.meta_learner.prediction_errors else 0.0,
                'num_predictions': len(self.meta_learner.prediction_errors)
            }
        
        return stats


class EnhancedQLearningAgent:
    """Q-Learning agent enhanced with intrinsic motivation"""
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.1, 
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1,
                 intrinsic_config: Optional[IntrinsicRewardConfig] = None):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Q-table
        self.q_table = np.zeros((state_size, action_size))
        
        # Intrinsic motivation
        if intrinsic_config:
            self.intrinsic_framework = IntrinsicMotivationFramework(intrinsic_config)
        else:
            self.intrinsic_framework = None
        
        # Performance tracking
        self.performance_history = deque(maxlen=50)
        
    def choose_action(self, state: int) -> int:
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int,
              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Update Q-table with intrinsic motivation"""
        
        total_reward = reward
        intrinsic_breakdown = {}
        
        # Add intrinsic reward if framework is available
        if self.intrinsic_framework and context:
            intrinsic_reward, intrinsic_breakdown = self.intrinsic_framework.compute_intrinsic_reward(
                state, action, next_state, context
            )
            total_reward += intrinsic_reward
            
            # Update intrinsic motivation modules
            experience = {
                'state': state,
                'action': action,
                'next_state': next_state,
                'reward': reward,
                'intrinsic_reward': intrinsic_reward,
                **context
            }
            self.intrinsic_framework.update_modules(experience)
        
        # Q-learning update
        current_q = self.q_table[state, action]
        next_max_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (total_reward + self.discount_factor * next_max_q - current_q)
        self.q_table[state, action] = new_q
        
        # Track performance
        self.performance_history.append(total_reward)
        
        return {
            'extrinsic_reward': reward,
            'intrinsic_breakdown': intrinsic_breakdown,
            'total_reward': total_reward,
            'q_value': new_q
        }
    
    def get_intrinsic_statistics(self) -> Dict[str, Any]:
        """Get intrinsic motivation statistics"""
        if self.intrinsic_framework:
            return self.intrinsic_framework.get_statistics()
        return {}


# Example usage and testing
def create_enhanced_agent_experiment():
    """Create example experiment with intrinsic motivation"""
    
    # Configure intrinsic motivation
    intrinsic_config = IntrinsicRewardConfig(
        curiosity_weight=0.1,
        information_gain_weight=0.05,
        insight_discovery_reward=1.0,
        adaptive_scheduling=True,
        meta_learning_enabled=True
    )
    
    # Create enhanced agent
    agent = EnhancedQLearningAgent(
        state_size=100,
        action_size=4,
        intrinsic_config=intrinsic_config
    )
    
    # Simulate training episode
    episode_rewards = []
    for episode in range(100):
        state = random.randint(0, 99)
        total_episode_reward = 0
        
        for step in range(50):
            action = agent.choose_action(state)
            next_state = random.randint(0, 99)
            extrinsic_reward = random.random() - 0.5
            
            # Create context for intrinsic motivation
            context = {
                'episode': episode,
                'step': step,
                'insight_detected': random.random() < 0.1,  # 10% chance of insight
                'insight_quality': random.random(),
                'recent_performance': list(agent.performance_history)
            }
            
            # Update agent
            update_info = agent.update(state, action, extrinsic_reward, next_state, context)
            total_episode_reward += update_info['total_reward']
            state = next_state
            
        episode_rewards.append(total_episode_reward)
        
        # Decay epsilon
        agent.epsilon *= 0.995
    
    # Get statistics
    stats = agent.get_intrinsic_statistics()
    
    return {
        'episode_rewards': episode_rewards,
        'intrinsic_stats': stats,
        'final_q_table': agent.q_table
    }


if __name__ == "__main__":
    # Run example experiment
    results = create_enhanced_agent_experiment()
    print("Intrinsic Motivation Framework Test Results:")
    print(f"Total Episodes: 100")
    print(f"Final Episode Reward: {results['episode_rewards'][-1]:.3f}")
    print(f"Total Intrinsic Reward: {results['intrinsic_stats']['total_intrinsic_reward']:.3f}")
    print(f"Number of Insights: {results['intrinsic_stats']['num_insights']}")
