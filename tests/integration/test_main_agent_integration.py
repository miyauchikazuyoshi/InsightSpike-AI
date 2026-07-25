"""Integration tests for MainAgent with Pydantic configuration."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from insightspike.config.models import InsightSpikeConfig
from insightspike.config.presets import ConfigPresets
from insightspike.core.base.datastore import DataStore
from insightspike.core.episode import Episode
from insightspike.implementations.agents.main_agent import CycleResult, MainAgent


@pytest.fixture
def test_config():
    """Create test configuration."""
    return ConfigPresets.development()


@pytest.fixture
def mock_datastore():
    """Create a mock datastore."""
    datastore = Mock(spec=DataStore)
    datastore.load_episodes.return_value = []
    datastore.save_episodes.return_value = True
    datastore.replace_episodes.return_value = True
    datastore.load_graph.return_value = None
    datastore.save_graph.return_value = True
    return datastore


@pytest.fixture
def agent(test_config, mock_datastore):
    """Create a MainAgent instance for testing."""
    return MainAgent(config=test_config, datastore=mock_datastore)


class TestMainAgentInitialization:
    """Test MainAgent initialization with Pydantic config."""

    def test_init_with_pydantic_config(self, test_config, mock_datastore):
        """Test agent initializes with Pydantic config."""
        agent = MainAgent(config=test_config, datastore=mock_datastore)

        assert agent is not None
        assert agent.config == test_config
        assert agent.is_pydantic_config is True
        assert agent.datastore == mock_datastore

    def test_init_requires_config(self, mock_datastore):
        """Test agent requires config parameter."""
        with pytest.raises(ValueError, match="Config must be provided"):
            MainAgent(config=None, datastore=mock_datastore)

    def test_initialize_components(self, agent):
        """Test component initialization."""
        result = agent.initialize()

        assert result is True
        assert agent._initialized is True

        # Check components are created
        assert agent.l1_error_monitor is not None
        assert agent.l2_memory is not None
        assert agent.l4_llm is not None

    def test_config_determines_memory_dimension(self, mock_datastore):
        """Test that config properly sets memory dimension."""
        # Custom config with different dimension
        config = InsightSpikeConfig(
            embedding={
                "model_name": "sentence-transformers/all-mpnet-base-v2",
                "dimension": 768,
            }
        )
        agent = MainAgent(config=config, datastore=mock_datastore)

        assert agent.l2_memory.config.embedding_dim == 768


class TestMainAgentProcessing:
    """Test MainAgent question processing."""

    def test_process_simple_question(self, agent):
        """Test processing a simple question."""
        agent.initialize()

        result = agent.process_question("What is InsightSpike?", max_cycles=1)

        assert isinstance(result, CycleResult)
        assert result.question == "What is InsightSpike?"
        assert result.success is True
        assert isinstance(result.response, str)
        assert result.cycle_number >= 1

    def test_process_with_multiple_cycles(self, agent):
        """Test processing with multiple reasoning cycles."""
        agent.initialize()

        # Add some initial knowledge
        agent.l2_memory.store_episode("InsightSpike is an AI system.", c_value=0.8)
        agent.l2_memory.store_episode(
            "It discovers insights through synthesis.", c_value=0.9
        )

        result = agent.process_question(
            "How does InsightSpike work?", max_cycles=3, verbose=True
        )

        assert result.success is True
        assert len(result.retrieved_documents) >= 0  # May retrieve the stored episodes
        assert result.reasoning_quality >= 0.0

    def test_spike_detection(self, agent, monkeypatch):
        """Test insight spike detection."""
        agent.initialize()

        # Add knowledge that should trigger spike when connected
        agent.l2_memory.store_episode("System A operates independently.", c_value=0.7)
        agent.l2_memory.store_episode("System B operates independently.", c_value=0.7)
        agent.l2_memory.store_episode("A and B together create emergence.", c_value=0.9)

        # The global test harness defaults to Lite mode. Replace only the
        # optional L3 boundary so this contract is independent of PyG.
        monkeypatch.setenv("INSIGHTSPIKE_LITE_MODE", "0")
        monkeypatch.setenv("INSIGHTSPIKE_MIN_IMPORT", "0")
        agent.l3_graph = Mock()
        agent.l3_graph.analyze_documents.return_value = {
            "graph": Mock(),
            "metrics": {"delta_ged": -0.8, "delta_ig": 0.5},
            "conflicts": {"total": 0.1},
            "reward": {"total": 0.9},
            "spike_detected": True,
            "graph_features": None,
            "reasoning_quality": 0.95,
        }

        result = agent.process_question(
            "What happens when A and B integrate?", max_cycles=2
        )

        assert result.spike_detected is True
        assert result.graph_analysis["spike_detected"] is True

    def test_error_handling(self, agent):
        """Test error handling during processing."""
        agent.initialize()

        # Mock memory search to raise error
        with patch.object(
            agent.l2_memory, "search_episodes", side_effect=Exception("Search error")
        ):
            result = agent.process_question("Test question", max_cycles=1)

            # Should still return a result, but with error state
            assert isinstance(result, CycleResult)
            assert result.success is True  # Graceful degradation


class TestMainAgentLearning:
    """Test MainAgent learning capabilities."""

    def test_add_single_document(self, agent):
        """Test adding a single document."""
        agent.initialize()

        result = agent.add_document(
            {
                "text": "InsightSpike uses graph-based reasoning.",
                "metadata": {"source": "test"},
            }
        )

        assert result is True
        assert agent.l2_memory.get_memory_stats()["total_episodes"] > 0
        episode = agent.l2_memory.episodes[-1]
        assert episode.text == "InsightSpike uses graph-based reasoning."
        assert episode.metadata == {"source": "test"}

    def test_learn_from_text(self, agent):
        """Test learning from text content."""
        agent.initialize()

        text_content = """
        InsightSpike is an AI system.
        It uses multiple layers of processing.
        Graph reasoning helps detect insights.
        """

        result = agent.learn(text_content)

        assert result["success"] is True
        assert result["episodes_added"] == 1
        # Current L3.update_graph is an acknowledged no-op; a successful call
        # must not be reported as a completed structural update.
        assert result["graph_updated"] is False

    def test_batch_learning(self, agent):
        """Test learning from multiple documents."""
        agent.initialize()

        documents = [
            {"text": "First concept about systems.", "metadata": {"id": 1}},
            {"text": "Second concept about integration.", "metadata": {"id": 2}},
            {"text": "Third concept about emergence.", "metadata": {"id": 3}},
        ]

        for doc in documents:
            agent.add_document(doc)

        stats = agent.get_stats()
        assert stats["memory_stats"]["total_episodes"] >= 3


class TestMainAgentStatePersistence:
    """Test MainAgent state persistence."""

    def test_save_state(self, agent, mock_datastore):
        """Test saving agent state."""
        agent.initialize()

        # Add some data
        agent.l2_memory.store_episode("Test episode", c_value=0.8)
        agent.l3_graph.previous_graph = {"nodes": [], "edges": []}

        # Save state
        result = agent.save_state()

        assert result is True
        mock_datastore.replace_episodes.assert_called()
        mock_datastore.save_graph.assert_called()

        records = mock_datastore.replace_episodes.call_args.args[0]
        assert records[0]["text"] == "Test episode"
        assert "metadata" in records[0]
        assert "episode_type" in records[0]
        assert "selection_count" in records[0]
        assert "creation_time" in records[0]

    def test_load_state(self, agent, mock_datastore):
        """Test loading agent state."""
        # Mock loaded data
        mock_episodes = [
            Episode(
                text="Loaded episode 1",
                vec=np.random.randn(384),
                c=0.7,
                metadata={"source": "object"},
            ),
            {
                "text": "Loaded episode 2",
                "vec": np.random.randn(384),
                "c": 0.8,
                "metadata": {"source": "mapping"},
            },
        ]
        mock_datastore.load_episodes.return_value = mock_episodes

        agent.initialize()
        result = agent.load_state()

        assert result is True
        mock_datastore.load_episodes.assert_called()

        # Check episodes were loaded
        stats = agent.get_stats()
        assert stats["memory_stats"]["total_episodes"] == 2
        assert [episode.c for episode in agent.l2_memory.episodes] == pytest.approx(
            [0.7, 0.8]
        )
        assert agent.l2_memory.episodes[0].metadata == {"source": "object"}
        assert agent.l2_memory.episodes[1].metadata == {"source": "mapping"}


class TestMainAgentWithDifferentConfigs:
    """Test MainAgent with different configuration presets."""

    def test_with_experiment_config(self, mock_datastore):
        """Test agent with experiment configuration."""
        config = ConfigPresets.experiment()
        agent = MainAgent(config=config, datastore=mock_datastore)
        agent.initialize()

        assert agent.config.environment == "experiment"
        assert agent.config.llm.provider == "local"
        assert agent.config.memory.max_episodes == 2000

        # Process a question
        result = agent.process_question("Test question", max_cycles=1)
        assert result.success is True

    def test_with_production_config(self, mock_datastore):
        """Test agent with production configuration."""
        config = ConfigPresets.production()
        agent = MainAgent(config=config, datastore=mock_datastore)

        # L4 is intentionally resolved lazily during initialize().
        provider = Mock(initialized=False)
        provider.initialize.return_value = True
        with patch(
            "insightspike.implementations.agents.main_agent.get_llm_provider",
            return_value=provider,
        ):
            assert agent.initialize() is True

            assert agent.config.environment == "production"
            assert agent.config.llm.provider == "openai"
            assert agent.config.memory.max_episodes == 5000
            assert agent.l4_llm is provider

    def test_with_custom_config(self, mock_datastore):
        """Test agent with custom configuration."""
        config = InsightSpikeConfig(
            environment="custom",
            llm={"provider": "mock", "temperature": 0.5},
            memory={"max_episodes": 500, "max_retrieved_docs": 5},
            embedding={"dimension": 384},
        )

        agent = MainAgent(config=config, datastore=mock_datastore)
        agent.initialize()

        assert agent.config.environment == "custom"
        assert agent.config.llm.temperature == 0.5
        assert agent.config.memory.max_retrieved_docs == 5


class TestMainAgentInsights:
    """Test MainAgent insight discovery features."""

    def test_get_insights(self, agent):
        """Test retrieving discovered insights."""
        agent.initialize()

        insights = [
            SimpleNamespace(
                id="insight-1",
                text="Important insight about emergence",
                discovery_context="What emerges?",
                generated_at=2.0,
                quality_score=0.95,
                relationship_type="synthetic",
            ),
            SimpleNamespace(
                id="insight-2",
                text="Another key insight",
                discovery_context="What integrates?",
                generated_at=1.0,
                quality_score=0.9,
                relationship_type="structural",
            ),
        ]
        agent.insight_registry = Mock()
        agent.insight_registry.insights = {
            insight.id: insight for insight in insights
        }
        agent.insight_registry.get_recent_insights.return_value = insights

        result = agent.get_insights(limit=2)

        assert result["total_insights"] == 2
        assert len(result["recent_insights"]) == 2
        assert result["recent_insights"][0]["answer"] == insights[0].text
        assert result["categories"] == ["structural", "synthetic"]

    def test_search_insights(self, agent):
        """Test searching for specific insights."""
        agent.initialize()

        insight = SimpleNamespace(
            text="Emergence occurs in complex systems",
            discovery_context="What is emergence?",
            generated_at=1.0,
            quality_score=0.9,
            relationship_type="synthetic",
        )
        agent.insight_registry = Mock()
        agent.insight_registry.search_insights_by_concept.return_value = [insight]

        results = agent.search_insights("emergence", limit=5)

        assert isinstance(results, list)
        assert any("emergence" in r.get("answer", "").lower() for r in results)
        agent.insight_registry.search_insights_by_concept.assert_called_once_with(
            "emergence",
            limit=5,
        )


class TestMemoryIntegration:
    """Test memory system integration."""

    def test_memory_retrieval_in_processing(self, agent):
        """Test that memory is properly retrieved during question processing."""
        agent.initialize()

        # Pre-populate memory
        agent.l2_memory.store_episode(
            "InsightSpike is an AI system for insights.", c_value=0.8
        )
        agent.l2_memory.store_episode("It uses graph-based reasoning.", c_value=0.85)

        result = agent.process_question("What is InsightSpike?", max_cycles=1)

        # Should retrieve relevant documents
        assert len(result.retrieved_documents) > 0
        # Retrieved docs should be relevant
        assert any(
            "InsightSpike" in doc.get("text", "") for doc in result.retrieved_documents
        )

    def test_memory_update_after_processing(self, agent):
        """Test that memory is updated after processing questions."""
        agent.initialize()

        initial_count = agent.l2_memory.get_memory_stats()["total_episodes"]

        # Process a question
        result = agent.process_question("What is emergence?", max_cycles=1)

        # Memory should be updated with Q&A
        final_count = agent.l2_memory.get_memory_stats()["total_episodes"]
        assert final_count > initial_count


class TestConvergenceDetection:
    """Test reasoning convergence detection."""

    def test_quality_based_convergence(self, agent):
        """Test convergence based on high reasoning quality."""
        agent.initialize()

        # Mock high quality reasoning
        with patch.object(agent, "_calculate_reasoning_quality", return_value=0.85):
            result = agent.process_question("Test question", max_cycles=5)

            # Should converge early due to high quality
            assert result.reasoning_quality > 0.8
            assert result.success is True

    def test_spike_based_convergence(self, agent, monkeypatch):
        """Test convergence when spike is detected."""
        agent.initialize()

        monkeypatch.setenv("INSIGHTSPIKE_LITE_MODE", "0")
        monkeypatch.setenv("INSIGHTSPIKE_MIN_IMPORT", "0")
        agent.l3_graph = Mock()
        agent.l3_graph.analyze_documents.return_value = {
            "graph": Mock(),
            "metrics": {"delta_ged": -0.9, "delta_ig": 0.6},
            "conflicts": {"total": 0.05},
            "reward": {"total": 0.95},
            "spike_detected": True,
            "graph_features": None,
            "reasoning_quality": 0.9,
        }

        result = agent.process_question("Test question", max_cycles=10)

        # Should converge due to spike detection
        assert result.spike_detected is True
        # Should have stopped before max_cycles
        assert agent.l3_graph.analyze_documents.call_count < 10
