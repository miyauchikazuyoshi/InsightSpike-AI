"""
Unit tests for generic agent implementation
"""
import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch
from collections import deque

from insightspike.core.agents.generic_agent import (
    GenericMemoryManager, GenericReasoner, SimpleInsightDetector, 
    GenericInsightSpikeAgent
)
from insightspike.core.interfaces.generic_interfaces import (
    EnvironmentState, InsightMoment, TaskType
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
        
        state = EnvironmentState(observation=np.array([1, 2, 3]))
        next_state = EnvironmentState(observation=np.array([4, 5, 6]))
        
        manager.store_experience(
            state=state,
            action=0,
            reward=1.0,
            next_state=next_state
        )
        
        assert len(manager.experiences) == 1
        assert manager.experiences[0]['state'] == state
        assert manager.experiences[0]['action'] == 0
        assert manager.experiences[0]['reward'] == 1.0
    
    def test_store_experience_with_insight(self):
        """Test storing experience with insight."""
        manager = GenericMemoryManager()
        
        state = EnvironmentState(observation=np.array([1, 2, 3]))
        insight = InsightMoment(
            value=2.0,
            state=state,
            description="Test insight"
        )
        
        manager.store_experience(
            state=state,
            action=1,
            reward=0.5,
            next_state=state,
            insight=insight
        )
        
        assert len(manager.insights) == 1
        assert manager.insights[0] == insight
    
    def test_retrieve_recent(self):
        """Test retrieving recent experiences."""
        manager = GenericMemoryManager()
        
        # Store multiple experiences
        for i in range(10):
            state = EnvironmentState(observation=np.array([i]))
            manager.store_experience(state, i, i/10, state)
        
        recent = manager.retrieve_recent(n=5)
        assert len(recent) == 5
        assert recent[0]['timestamp'] == 5  # Most recent first
    
    def test_retrieve_insights(self):
        """Test retrieving insights."""
        manager = GenericMemoryManager()
        
        # Add insights
        for i in range(3):
            insight = InsightMoment(
                value=i,
                state=EnvironmentState(observation=np.array([i])),
                description=f"Insight {i}"
            )
            manager.store_experience(
                state=insight.state,
                action=0,
                reward=0,
                next_state=insight.state,
                insight=insight
            )
        
        insights = manager.retrieve_insights(n=2)
        assert len(insights) == 2
        assert insights[0].value == 2  # Highest value first
    
    def test_max_capacity(self):
        """Test max capacity enforcement."""
        manager = GenericMemoryManager(max_capacity=3)
        
        for i in range(5):
            state = EnvironmentState(observation=np.array([i]))
            manager.store_experience(state, 0, 0, state)
        
        assert len(manager.experiences) == 3
        # Should keep most recent
        assert manager.experiences[-1]['state'].observation[0] == 4
    
    def test_get_statistics(self):
        """Test getting memory statistics."""
        manager = GenericMemoryManager()
        
        # Add some experiences and insights
        for i in range(5):
            state = EnvironmentState(observation=np.array([i]))
            insight = InsightMoment(value=i, state=state) if i % 2 == 0 else None
            manager.store_experience(state, 0, i, state, insight)
        
        stats = manager.get_statistics()
        assert stats['total_experiences'] == 5
        assert stats['total_insights'] == 3
        assert stats['avg_reward'] == 2.0


class TestGenericReasoner:
    """Test GenericReasoner functionality."""
    
    def test_init(self):
        """Test reasoner initialization."""
        reasoner = GenericReasoner(strategy='exploration')
        assert reasoner.strategy == 'exploration'
        assert reasoner.epsilon == 0.2
    
    def test_reason_exploration(self):
        """Test exploration reasoning."""
        reasoner = GenericReasoner(strategy='exploration', epsilon=1.0)
        state = EnvironmentState(observation=np.array([1, 2, 3]))
        memory = GenericMemoryManager()
        
        with patch('numpy.random.random', return_value=0.5):
            with patch('numpy.random.choice', return_value=2):
                action = reasoner.reason(state, memory, task_type=TaskType.EXPLORATION)
                assert action == 2
    
    def test_reason_exploitation(self):
        """Test exploitation reasoning."""
        reasoner = GenericReasoner(strategy='exploitation')
        state = EnvironmentState(observation=np.array([1, 2, 3]))
        memory = GenericMemoryManager()
        
        # Add experiences with different rewards
        for i in range(3):
            memory.store_experience(
                state=state,
                action=i,
                reward=i * 10,  # Action 2 has highest reward
                next_state=state
            )
        
        action = reasoner.reason(state, memory, task_type=TaskType.OPTIMIZATION)
        assert action == 2  # Should choose highest reward action
    
    def test_reason_with_insights(self):
        """Test reasoning with insights."""
        reasoner = GenericReasoner()
        state = EnvironmentState(observation=np.array([1, 2, 3]))
        memory = GenericMemoryManager()
        
        # Add insight
        insight = InsightMoment(
            value=10.0,
            state=state,
            action=1,
            description="Good action"
        )
        memory.store_experience(state, 1, 5.0, state, insight)
        
        # Should prefer action from insight
        action = reasoner.reason(state, memory, task_type=TaskType.DISCOVERY)
        assert action == 1
    
    def test_extract_features(self):
        """Test feature extraction."""
        reasoner = GenericReasoner()
        state = EnvironmentState(
            observation=np.array([1, 2, 3]),
            task_info={'difficulty': 0.5}
        )
        
        features = reasoner.extract_features(state)
        assert 'observation_mean' in features
        assert 'observation_std' in features
        assert features['task_difficulty'] == 0.5


class TestSimpleInsightDetector:
    """Test SimpleInsightDetector functionality."""
    
    def test_init(self):
        """Test detector initialization."""
        detector = SimpleInsightDetector(threshold=2.0)
        assert detector.threshold == 2.0
        assert detector.baseline == 0.0
    
    def test_detect_insight_above_threshold(self):
        """Test insight detection above threshold."""
        detector = SimpleInsightDetector(threshold=2.0)
        state = EnvironmentState(observation=np.array([1]))
        
        # High reward should trigger insight
        insight = detector.detect_insight(state, 0, 5.0, state)
        assert insight is not None
        assert insight.value == 5.0
        assert "High reward" in insight.description
    
    def test_detect_insight_below_threshold(self):
        """Test no insight below threshold."""
        detector = SimpleInsightDetector(threshold=2.0)
        state = EnvironmentState(observation=np.array([1]))
        
        # Low reward should not trigger insight
        insight = detector.detect_insight(state, 0, 1.0, state)
        assert insight is None
    
    def test_baseline_update(self):
        """Test baseline updating."""
        detector = SimpleInsightDetector()
        
        # Process several rewards
        for reward in [1.0, 2.0, 3.0]:
            state = EnvironmentState(observation=np.array([1]))
            detector.detect_insight(state, 0, reward, state)
        
        # Baseline should be updated
        assert detector.baseline > 0
        assert detector.baseline < 3.0


class TestGenericInsightSpikeAgent:
    """Test GenericInsightSpikeAgent functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.env = Mock()
        self.env.reset.return_value = EnvironmentState(observation=np.array([0]))
        self.env.get_action_space.return_value = list(range(4))
        
        self.state_encoder = Mock()
        self.state_encoder.encode.return_value = np.array([0, 1])
        
        self.reward_normalizer = Mock()
        self.reward_normalizer.normalize.return_value = 1.0
    
    def test_init(self):
        """Test agent initialization."""
        agent = GenericInsightSpikeAgent(
            environment=self.env,
            state_encoder=self.state_encoder,
            reward_normalizer=self.reward_normalizer
        )
        
        assert agent.environment == self.env
        assert agent.state_encoder == self.state_encoder
        assert isinstance(agent.memory, GenericMemoryManager)
        assert isinstance(agent.reasoner, GenericReasoner)
        assert isinstance(agent.insight_detector, SimpleInsightDetector)
    
    def test_act(self):
        """Test agent action selection."""
        agent = GenericInsightSpikeAgent(
            environment=self.env,
            state_encoder=self.state_encoder
        )
        
        state = EnvironmentState(observation=np.array([1, 2]))
        agent.reasoner.reason = Mock(return_value=2)
        
        action = agent.act(state, task_type=TaskType.EXPLORATION)
        assert action == 2
        agent.reasoner.reason.assert_called_once()
    
    def test_step(self):
        """Test agent step execution."""
        agent = GenericInsightSpikeAgent(
            environment=self.env,
            state_encoder=self.state_encoder,
            reward_normalizer=self.reward_normalizer
        )
        
        # Mock environment step
        next_state = EnvironmentState(observation=np.array([2]))
        self.env.step.return_value = (next_state, 5.0, False, {})
        
        # Mock action selection
        agent.act = Mock(return_value=1)
        
        # Execute step
        state = EnvironmentState(observation=np.array([1]))
        result = agent.step(state, task_type=TaskType.EXPLORATION)
        
        assert result['next_state'] == next_state
        assert result['reward'] == 1.0  # Normalized
        assert result['done'] is False
        assert 'insight' in result
    
    def test_run_episode(self):
        """Test running full episode."""
        agent = GenericInsightSpikeAgent(
            environment=self.env,
            state_encoder=self.state_encoder,
            reward_normalizer=self.reward_normalizer
        )
        
        # Mock environment behavior
        states = [
            EnvironmentState(observation=np.array([i])) 
            for i in range(3)
        ]
        self.env.reset.return_value = states[0]
        self.env.step.side_effect = [
            (states[1], 1.0, False, {}),
            (states[2], 10.0, True, {})  # High reward at end
        ]
        
        # Mock action selection
        agent.act = Mock(return_value=0)
        
        # Run episode
        total_reward, insights = agent.run_episode(
            max_steps=10,
            task_type=TaskType.EXPLORATION
        )
        
        assert total_reward == 2.0  # Both rewards normalized to 1.0
        assert len(insights) >= 1  # Should detect high reward insight
    
    def test_get_statistics(self):
        """Test getting agent statistics."""
        agent = GenericInsightSpikeAgent(
            environment=self.env,
            state_encoder=self.state_encoder
        )
        
        # Add some data
        for i in range(3):
            state = EnvironmentState(observation=np.array([i]))
            agent.memory.store_experience(state, i, i, state)
        
        stats = agent.get_statistics()
        assert 'episodes_completed' in stats
        assert 'memory_stats' in stats
        assert stats['memory_stats']['total_experiences'] == 3