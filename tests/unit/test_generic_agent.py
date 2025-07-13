"""
Unit tests for generic agent implementation
"""
import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch
from collections import deque

from insightspike.core.agents.generic_agent import (
    GenericMemoryManager,
    GenericReasoner,
    GenericInsightSpikeAgent,
)
from insightspike.core.interfaces.generic_interfaces import (
    EnvironmentState,
    InsightMoment,
    TaskType,
    ActionSpace,
    EnvironmentInterface,
    InsightDetectorInterface,
    StateEncoder,
    RewardNormalizer,
)


class TestGenericMemoryManager:
    """Test GenericMemoryManager functionality."""

    def test_init(self):
        """Test memory manager initialization."""
        manager = GenericMemoryManager(max_capacity=100)
        assert manager.max_capacity == 100
        assert len(manager.experiences) == 0
        assert len(manager.insights) == 0

    def test_store_experience(self):
        """Test storing experiences."""
        manager = GenericMemoryManager()

        state = EnvironmentState(
            state_data=np.array([1, 2, 3]),
            environment_type="test",
            task_type=TaskType.CUSTOM,
        )
        next_state = EnvironmentState(
            state_data=np.array([4, 5, 6]),
            environment_type="test",
            task_type=TaskType.CUSTOM,
        )

        manager.store_experience(
            state=state, action=0, reward=1.0, next_state=next_state
        )

        assert len(manager.experiences) == 1
        assert manager.experiences[0]["state"] == state
        assert manager.experiences[0]["action"] == 0
        assert manager.experiences[0]["reward"] == 1.0

    def test_store_experience_with_insight(self):
        """Test storing experience with insight."""
        manager = GenericMemoryManager()

        state = EnvironmentState(
            state_data=np.array([1, 2, 3]),
            environment_type="test",
            task_type=TaskType.CUSTOM,
        )
        insight = InsightMoment(
            step=0,
            episode=0,
            insight_type="test_insight",
            description="Test insight",
            confidence=0.9,
            dged_value=0.1,
            dig_value=0.2,
        )

        manager.store_experience(
            state=state, action=1, reward=0.5, next_state=state, insight=insight
        )

        assert len(manager.insights) == 1
        assert manager.insights[0] == insight

    def test_retrieve_similar(self):
        """Test retrieving similar experiences."""
        manager = GenericMemoryManager()

        # Store multiple experiences
        for i in range(5):
            state = EnvironmentState(
                state_data=np.array([i]),
                environment_type="test",
                task_type=TaskType.CUSTOM,
            )
            manager.store_experience(state, i, i / 10, state)

        query_state = EnvironmentState(
            state_data=np.array([2]), environment_type="test", task_type=TaskType.CUSTOM
        )

        similar = manager.retrieve_similar(query_state, k=3)
        assert len(similar) <= 3
        assert all("state" in exp for exp in similar)

    def test_max_capacity(self):
        """Test max capacity enforcement."""
        manager = GenericMemoryManager(max_capacity=3)

        for i in range(5):
            state = EnvironmentState(
                state_data=np.array([i]),
                environment_type="test",
                task_type=TaskType.CUSTOM,
            )
            manager.store_experience(state, 0, 0, state)

        assert len(manager.experiences) == 3

    def test_get_memory_stats(self):
        """Test getting memory statistics."""
        manager = GenericMemoryManager()

        # Add some experiences and insights
        for i in range(5):
            state = EnvironmentState(
                state_data=np.array([i]),
                environment_type="test",
                task_type=TaskType.CUSTOM,
            )
            insight = (
                InsightMoment(
                    step=i,
                    episode=0,
                    insight_type="test",
                    description=f"Insight {i}",
                    confidence=0.9,
                    dged_value=0.1,
                    dig_value=0.2,
                )
                if i % 2 == 0
                else None
            )
            manager.store_experience(state, 0, i, state, insight)

        stats = manager.get_memory_stats()
        assert stats["total_experiences"] == 5
        assert stats["total_insights"] == 3
        assert "memory_utilization" in stats
        assert "insight_rate" in stats


class TestGenericReasoner:
    """Test GenericReasoner functionality."""

    def test_init(self):
        """Test reasoner initialization."""
        reasoner = GenericReasoner()
        assert reasoner is not None

    def test_analyze_insight_pattern(self):
        """Test insight pattern analysis."""
        reasoner = GenericReasoner()

        insights = [
            InsightMoment(
                step=i * 10,
                episode=0,
                insight_type="type_a" if i % 2 == 0 else "type_b",
                description=f"Insight {i}",
                confidence=0.9,
                dged_value=0.1,
                dig_value=0.2,
            )
            for i in range(5)
        ]

        analysis = reasoner.analyze_insight_pattern(insights)
        assert analysis["pattern_detected"] is True
        assert analysis["total_insights"] == 5
        assert "insight_types" in analysis
        assert "avg_interval" in analysis

    def test_analyze_empty_insights(self):
        """Test pattern analysis with no insights."""
        reasoner = GenericReasoner()
        analysis = reasoner.analyze_insight_pattern([])
        assert analysis["pattern_detected"] is False

    def test_predict_next_insight(self):
        """Test insight prediction."""
        reasoner = GenericReasoner()

        context = {"recent_rewards": [1.0, 2.0, 3.0], "exploration_ratio": 0.3}

        predictions = reasoner.predict_next_insight(context)
        assert isinstance(predictions, dict)
        assert "strategic_breakthrough" in predictions
        assert "exploration_insight" in predictions
        assert all(0 <= p <= 1 for p in predictions.values())


class TestGenericInsightSpikeAgent:
    """Test GenericInsightSpikeAgent functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock environment
        self.env = MagicMock(spec=EnvironmentInterface)
        self.env.get_action_space.return_value = ActionSpace(
            action_type="discrete", action_dim=4
        )
        self.env.reset.return_value = EnvironmentState(
            state_data=np.array([0, 0]),
            environment_type="test",
            task_type=TaskType.CUSTOM,
        )

        # Mock insight detector
        self.insight_detector = MagicMock(spec=InsightDetectorInterface)
        self.insight_detector.detect_insight.return_value = None

        # Mock state encoder
        self.state_encoder = MagicMock(spec=StateEncoder)
        self.state_encoder.encode_state.return_value = np.array([0, 1])

        # Mock reward normalizer
        self.reward_normalizer = MagicMock(spec=RewardNormalizer)
        self.reward_normalizer.normalize_reward.return_value = 1.0

    def test_init(self):
        """Test agent initialization."""
        agent = GenericInsightSpikeAgent(
            agent_id="test-agent",
            environment=self.env,
            insight_detector=self.insight_detector,
            state_encoder=self.state_encoder,
            reward_normalizer=self.reward_normalizer,
        )

        assert agent.agent_id == "test-agent"
        assert agent.environment == self.env
        assert isinstance(agent.memory, GenericMemoryManager)
        assert isinstance(agent.reasoner, GenericReasoner)

    def test_select_action(self):
        """Test action selection."""
        agent = GenericInsightSpikeAgent(
            agent_id="test-agent",
            environment=self.env,
            insight_detector=self.insight_detector,
            state_encoder=self.state_encoder,
            reward_normalizer=self.reward_normalizer,
        )

        state = EnvironmentState(
            state_data=np.array([1, 2]),
            environment_type="test",
            task_type=TaskType.CUSTOM,
        )

        action = agent.select_action(state)
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 4

    def test_update(self):
        """Test agent update."""
        agent = GenericInsightSpikeAgent(
            agent_id="test-agent",
            environment=self.env,
            insight_detector=self.insight_detector,
            state_encoder=self.state_encoder,
            reward_normalizer=self.reward_normalizer,
        )

        state = EnvironmentState(
            state_data=np.array([1, 2]),
            environment_type="test",
            task_type=TaskType.CUSTOM,
        )
        next_state = EnvironmentState(
            state_data=np.array([2, 3]),
            environment_type="test",
            task_type=TaskType.CUSTOM,
        )

        # Update agent
        agent.update(state, 1, 5.0, next_state, False)

        # Check memory was updated
        assert len(agent.memory.experiences) == 1
        assert len(agent.recent_rewards) == 1

    def test_update_with_insight(self):
        """Test agent update with insight detection."""
        # Create insight
        test_insight = InsightMoment(
            step=10,
            episode=1,
            insight_type="test_insight",
            description="Test insight detected",
            confidence=0.95,
            dged_value=0.1,
            dig_value=0.2,
        )

        # Setup insight detector to return insight
        self.insight_detector.detect_insight.return_value = test_insight

        agent = GenericInsightSpikeAgent(
            agent_id="test-agent",
            environment=self.env,
            insight_detector=self.insight_detector,
            state_encoder=self.state_encoder,
            reward_normalizer=self.reward_normalizer,
        )

        state = EnvironmentState(
            state_data=np.array([1, 2]),
            environment_type="test",
            task_type=TaskType.CUSTOM,
            step_count=10,
            episode_count=1,
        )

        # Update should detect insight
        agent.update(state, 1, 10.0, state, False)

        assert len(agent.insight_moments) == 1
        assert agent.insight_moments[0] == test_insight
        assert agent.insight_boost_duration > 0

    def test_train_episode(self):
        """Test training an episode."""
        # Setup environment behavior
        states = [
            EnvironmentState(
                state_data=np.array([i, i + 1]),
                environment_type="test",
                task_type=TaskType.CUSTOM,
            )
            for i in range(3)
        ]

        self.env.reset.return_value = states[0]
        self.env.step.side_effect = [
            (states[1], 1.0, False, {}),
            (states[2], 2.0, True, {}),  # Episode ends
        ]

        agent = GenericInsightSpikeAgent(
            agent_id="test-agent",
            environment=self.env,
            insight_detector=self.insight_detector,
            state_encoder=self.state_encoder,
            reward_normalizer=self.reward_normalizer,
        )

        # Train episode
        result = agent.train_episode()

        assert result["episode"] == 1
        assert result["steps"] == 2
        assert "reward" in result
        assert "insights" in result
        assert agent.episode_count == 1

    def test_get_performance_summary(self):
        """Test getting performance summary."""
        agent = GenericInsightSpikeAgent(
            agent_id="test-agent",
            environment=self.env,
            insight_detector=self.insight_detector,
            state_encoder=self.state_encoder,
            reward_normalizer=self.reward_normalizer,
        )

        # Add some data
        agent.episode_rewards = [10.0, 20.0, 15.0]
        agent.episode_steps = [100, 150, 120]

        summary = agent.get_performance_summary()
        assert "total_episodes" in summary
        assert "total_insights" in summary
        assert "avg_episode_reward" in summary
        assert "memory_stats" in summary
        assert "insight_analysis" in summary
