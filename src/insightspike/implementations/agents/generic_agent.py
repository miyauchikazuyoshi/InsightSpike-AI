"""
Generic InsightSpike Agent Implementation
========================================

A reusable InsightSpike agent that can work with any environment by using
the generic interfaces. This agent removes all hardcoded dependencies and
can be configured for different domains.
"""

import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np

from ...core.base.generic_interfaces import (
    EnvironmentInterface,
    EnvironmentState,
    GenericAgentInterface,
    InsightDetectorInterface,
    InsightMoment,
    MemoryManagerInterface,
    ReasonerInterface,
    RewardNormalizer,
    StateEncoder,
    TaskType,
)

logger = logging.getLogger(__name__)


class GenericMemoryManager(MemoryManagerInterface):
    """Generic memory manager implementation"""

    def __init__(self, max_capacity: int = 10000):
        self.max_capacity = max_capacity
        self.experiences = deque(maxlen=max_capacity)
        self.insights = []

    def store_experience(
        self,
        state: EnvironmentState,
        action: Any,
        reward: float,
        next_state: EnvironmentState,
        insight: Optional[InsightMoment] = None,
    ):
        """Store experience in memory"""
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "insight": insight,
            "timestamp": len(self.experiences),
        }

        self.experiences.append(experience)

        if insight:
            self.insights.append(insight)

    def retrieve_similar(
        self, query_state: EnvironmentState, k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve k most similar experiences (simple implementation)"""
        if not self.experiences:
            return []

        # Simple similarity based on state data comparison
        # In practice, you'd use more sophisticated similarity measures
        similarities = []

        for exp in self.experiences:
            # Basic similarity calculation
            similarity = self._calculate_similarity(query_state, exp["state"])
            similarities.append((similarity, exp))

        # Sort by similarity and return top k
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [exp for _, exp in similarities[:k]]

    def _calculate_similarity(
        self, state1: EnvironmentState, state2: EnvironmentState
    ) -> float:
        """Calculate similarity between states (basic implementation)"""
        # This is a placeholder - implement domain-specific similarity
        if state1.task_type != state2.task_type:
            return 0.0

        # For now, return random similarity - replace with actual calculation
        return np.random.random()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "total_experiences": len(self.experiences),
            "total_insights": len(self.insights),
            "memory_utilization": len(self.experiences) / self.max_capacity,
            "insight_rate": len(self.insights) / max(len(self.experiences), 1),
        }


class GenericReasoner(ReasonerInterface):
    """Generic reasoner for insight analysis"""

    def analyze_insight_pattern(self, insights: List[InsightMoment]) -> Dict[str, Any]:
        """Analyze patterns in insight sequences"""
        if not insights:
            return {"pattern_detected": False}

        # Basic pattern analysis
        insight_types = [insight.insight_type for insight in insights]
        type_counts = defaultdict(int)
        for itype in insight_types:
            type_counts[itype] += 1

        # Time intervals between insights
        intervals = []
        for i in range(1, len(insights)):
            interval = insights[i].step - insights[i - 1].step
            intervals.append(interval)

        return {
            "pattern_detected": True,
            "total_insights": len(insights),
            "insight_types": dict(type_counts),
            "avg_interval": np.mean(intervals) if intervals else 0,
            "insight_trend": "increasing"
            if len(insights) > 5 and len(insights[-3:]) > len(insights[-6:-3])
            else "stable",
        }

    def predict_next_insight(self, current_context: Dict[str, Any]) -> Dict[str, float]:
        """Predict likelihood of insight in different scenarios"""
        # Basic prediction based on context
        base_prob = 0.1

        # Adjust based on recent performance
        if "recent_rewards" in current_context:
            recent_performance = np.mean(current_context["recent_rewards"])
            if recent_performance > 0:
                base_prob *= 1.5

        # Adjust based on exploration
        if "exploration_ratio" in current_context:
            exploration = current_context["exploration_ratio"]
            if exploration < 0.5:  # Low exploration might lead to insights
                base_prob *= 1.3

        return {
            "strategic_breakthrough": base_prob * 0.4,
            "exploration_insight": base_prob * 0.3,
            "goal_discovery": base_prob * 0.2,
            "efficiency_insight": base_prob * 0.1,
        }


class GenericInsightSpikeAgent(GenericAgentInterface):
    """Generic InsightSpike agent that works with any environment"""

    def __init__(
        self,
        agent_id: str,
        environment: EnvironmentInterface,
        insight_detector: InsightDetectorInterface,
        state_encoder: StateEncoder,
        reward_normalizer: RewardNormalizer,
        config: Optional[Dict[str, Any]] = None,
    ):
        # Initialize memory and reasoner
        memory = GenericMemoryManager()
        reasoner = GenericReasoner()

        super().__init__(
            agent_id, environment, insight_detector, state_encoder, reward_normalizer
        )

        self.memory = memory
        self.reasoner = reasoner
        self.config = config or {}

        # Learning parameters (configurable)
        self.learning_rate = self.config.get("learning_rate", 0.1)
        self.exploration_rate = self.config.get("exploration_rate", 0.3)
        self.exploration_decay = self.config.get("exploration_decay", 0.995)
        self.min_exploration = self.config.get("min_exploration", 0.05)

        # Insight-adaptive parameters
        self.insight_boost_duration = 0
        self.base_learning_rate = self.learning_rate

        # Action-value function (generic)
        self.action_space = environment.get_action_space()
        if self.action_space.action_type == "discrete":
            self.q_values = defaultdict(lambda: np.zeros(self.action_space.action_dim))
        else:
            # For continuous actions, you'd use a function approximator
            raise NotImplementedError("Continuous action spaces not implemented yet")

        # Performance tracking
        self.recent_rewards = deque(maxlen=100)
        self.episode_rewards = []
        self.episode_steps = []

    def select_action(self, state: EnvironmentState) -> Any:
        """Select action using insight-informed policy"""
        # Get state encoding for action selection
        encoded_state = tuple(self.state_encoder.encode_state(state))

        # Insight-adaptive exploration
        current_exploration = self.exploration_rate
        if self.insight_boost_duration > 0:
            current_exploration *= 1.5  # More exploration after insights
            self.insight_boost_duration -= 1

        # Apply minimum exploration
        current_exploration = max(current_exploration, self.min_exploration)

        # Epsilon-greedy action selection
        if np.random.random() < current_exploration:
            # Random action
            return np.random.randint(self.action_space.action_dim)
        else:
            # Greedy action
            return np.argmax(self.q_values[encoded_state])

    def update(
        self,
        state: EnvironmentState,
        action: Any,
        reward: float,
        next_state: EnvironmentState,
        done: bool,
    ):
        """Update agent based on transition"""
        # Normalize reward
        context = {
            "exploration_ratio": next_state.metadata.get("exploration_ratio", 0)
            if next_state.metadata
            else 0
        }
        normalized_reward = self.reward_normalizer.normalize_reward(reward, context)

        # Store in memory
        self.memory.store_experience(state, action, normalized_reward, next_state)

        # Detect insights
        insight_context = {
            "distance_to_goal": next_state.metadata.get("distance_to_goal", 0)
            if next_state.metadata
            else 0,
            "exploration_ratio": context["exploration_ratio"],
            "episode": next_state.episode_count,
            "step": next_state.step_count,
        }

        insight = self.insight_detector.detect_insight(
            state, action, normalized_reward, next_state, insight_context
        )

        if insight:
            logger.info(f"🧠 INSIGHT DETECTED! {insight.description}")
            self.insight_moments.append(insight)
            self.insight_boost_duration = 10  # Boost learning for next 10 steps

            # Store insight in memory
            self.memory.store_experience(
                state, action, normalized_reward, next_state, insight
            )

        # Update Q-values
        self._update_q_values(state, action, normalized_reward, next_state, done)

        # Update tracking
        self.recent_rewards.append(normalized_reward)
        self.step_count += 1

    def _update_q_values(
        self,
        state: EnvironmentState,
        action: Any,
        reward: float,
        next_state: EnvironmentState,
        done: bool,
    ):
        """Update Q-values with insight-enhanced learning"""
        encoded_state = tuple(self.state_encoder.encode_state(state))
        encoded_next_state = tuple(self.state_encoder.encode_state(next_state))

        # Current Q-value
        current_q = self.q_values[encoded_state][action]

        # Target Q-value
        if done:
            target_q = reward
        else:
            target_q = reward + 0.95 * np.max(self.q_values[encoded_next_state])

        # Insight-enhanced learning rate
        effective_lr = self.learning_rate
        if self.insight_boost_duration > 0:
            effective_lr = min(self.base_learning_rate * 1.5, 0.5)  # Boost but cap

        # Q-learning update
        self.q_values[encoded_state][action] += effective_lr * (target_q - current_q)

    def train_episode(self) -> Dict[str, Any]:
        """Train for one episode"""
        state = self.environment.reset()
        episode_reward = 0
        episode_steps = 0
        insights_this_episode = 0

        while True:
            # Select and execute action
            action = self.select_action(state)
            next_state, reward, done, info = self.environment.step(action)

            # Count insights before update
            insights_before = len(self.insight_moments)

            # Update agent
            self.update(state, action, reward, next_state, done)

            # Count insights after update
            insights_after = len(self.insight_moments)
            insights_this_episode += insights_after - insights_before

            # Track performance
            episode_reward += reward
            episode_steps += 1
            state = next_state

            if done:
                break

        # Update episode tracking
        self.episode_count += 1
        self.episode_rewards.append(episode_reward)
        self.episode_steps.append(episode_steps)

        # Decay exploration
        self.exploration_rate = max(
            self.min_exploration, self.exploration_rate * self.exploration_decay
        )

        return {
            "episode": self.episode_count,
            "reward": episode_reward,
            "steps": episode_steps,
            "insights": insights_this_episode,
            "exploration_rate": self.exploration_rate,
            "total_insights": len(self.insight_moments),
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        base_summary = super().get_performance_summary()

        # Add agent-specific metrics
        base_summary.update(
            {
                "avg_episode_reward": np.mean(self.episode_rewards)
                if self.episode_rewards
                else 0.0,
                "avg_episode_steps": np.mean(self.episode_steps)
                if self.episode_steps
                else 0.0,
                "recent_performance": np.mean(list(self.recent_rewards)[-20:])
                if len(self.recent_rewards) >= 20
                else 0.0,
                "memory_stats": self.memory.get_memory_stats(),
                "insight_analysis": self.reasoner.analyze_insight_pattern(
                    self.insight_moments
                ),
            }
        )

        return base_summary

    def save_state(self, filepath: str):
        """Save agent state for later loading"""
        # Implementation for saving agent state
        # This would serialize Q-values, memory, etc.
        pass

    def load_state(self, filepath: str):
        """Load agent state from file"""
        # Implementation for loading agent state
        pass


# Export the generic agent
__all__ = ["GenericMemoryManager", "GenericReasoner", "GenericInsightSpikeAgent"]
