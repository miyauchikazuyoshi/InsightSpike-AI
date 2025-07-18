"""Tests for the spike CLI with Pydantic configuration."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from typer.testing import CliRunner

from insightspike.cli.spike import DependencyFactory, app
from insightspike.config.models import InsightSpikeConfig
from insightspike.config.presets import ConfigPresets
from insightspike.core.base.datastore import DataStore
from insightspike.implementations.agents.main_agent import CycleResult


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_datastore():
    """Create a mock datastore."""
    return Mock(spec=DataStore)


@pytest.fixture
def mock_factory(mock_datastore):
    """Create a mock dependency factory."""
    config = ConfigPresets.development()
    return DependencyFactory(config, mock_datastore)


@pytest.fixture
def mock_agent():
    """Create a mock agent with predefined responses."""
    agent = Mock()
    agent.initialize.return_value = True
    agent.load_state.return_value = None

    # Mock process_question response
    agent.process_question.return_value = CycleResult(
        question="What is InsightSpike?",
        retrieved_documents=[],
        graph_analysis={"spike_detected": False, "metrics": {}},
        response="InsightSpike is an AI system for discovering insights through knowledge synthesis.",
        reasoning_quality=0.85,
        spike_detected=False,
        error_state={},
        cycle_number=1,
        success=True,
    )

    # Mock get_stats response
    agent.get_stats.return_value = {
        "initialized": True,
        "total_cycles": 10,
        "reasoning_history_length": 5,
        "average_quality": 0.75,
        "memory_stats": {"total_episodes": 100, "total_documents": 50},
    }

    return agent


class TestSpikeQuery:
    """Test the query command."""

    def test_query_basic(self, runner, mock_factory, mock_agent):
        """Test basic query functionality."""
        with patch.object(mock_factory, "get_agent", return_value=mock_agent):
            result = runner.invoke(
                app, ["query", "What is InsightSpike?"], obj=mock_factory
            )

            assert result.exit_code == 0
            assert "What is InsightSpike?" in result.stdout
            assert "InsightSpike is an AI system" in result.stdout
            mock_agent.process_question.assert_called_once()

    def test_query_with_preset(self, runner, mock_factory, mock_agent):
        """Test query with different preset."""
        with patch.object(
            mock_factory, "get_agent", return_value=mock_agent
        ) as mock_get:
            result = runner.invoke(
                app,
                ["query", "Test question", "--preset", "experiment"],
                obj=mock_factory,
            )

            assert result.exit_code == 0
            mock_get.assert_called_with("experiment")

    def test_query_verbose(self, runner, mock_factory, mock_agent):
        """Test query with verbose output."""
        with patch.object(mock_factory, "get_agent", return_value=mock_agent):
            result = runner.invoke(
                app, ["query", "Test question", "--verbose"], obj=mock_factory
            )

            assert result.exit_code == 0
            # Verbose output should include quality score
            assert "Quality score:" in result.stdout or "0.85" in result.stdout

    def test_query_with_spike_detection(self, runner, mock_factory, mock_agent):
        """Test query that detects an insight spike."""
        # Modify agent response to indicate spike
        mock_agent.process_question.return_value = CycleResult(
            question="What happens when systems integrate?",
            retrieved_documents=[],
            graph_analysis={"spike_detected": True, "metrics": {"delta_ged": -0.8}},
            response="Integration creates emergent properties.",
            reasoning_quality=0.95,
            spike_detected=True,
            error_state={},
            cycle_number=2,
            success=True,
        )

        with patch.object(mock_factory, "get_agent", return_value=mock_agent):
            result = runner.invoke(
                app, ["query", "What happens when systems integrate?"], obj=mock_factory
            )

            assert result.exit_code == 0
            assert "INSIGHT SPIKE DETECTED" in result.stdout


class TestSpikeEmbed:
    """Test the embed command."""

    def test_embed_text_file(self, runner, mock_factory, mock_agent):
        """Test embedding a text file."""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is test content.\nIt has multiple lines.")
            temp_path = Path(f.name)

        try:
            mock_agent.learn.return_value = {
                "success": True,
                "episodes_added": 2,
                "graph_updated": True,
                "insights": [],
            }

            with patch.object(mock_factory, "get_agent", return_value=mock_agent):
                result = runner.invoke(app, ["embed", str(temp_path)], obj=mock_factory)

                assert result.exit_code == 0
                assert "Successfully embedded" in result.stdout
                mock_agent.learn.assert_called_once()
        finally:
            temp_path.unlink()

    def test_embed_directory(self, runner, mock_factory, mock_agent):
        """Test embedding all text files in a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            dir_path = Path(temp_dir)
            (dir_path / "file1.txt").write_text("Content 1")
            (dir_path / "file2.txt").write_text("Content 2")
            (dir_path / "ignored.pdf").write_text("Should be ignored")

            mock_agent.learn.return_value = {
                "success": True,
                "episodes_added": 2,
                "graph_updated": True,
                "insights": [],
            }

            with patch.object(mock_factory, "get_agent", return_value=mock_agent):
                result = runner.invoke(app, ["embed", str(dir_path)], obj=mock_factory)

                assert result.exit_code == 0
                assert "2 files" in result.stdout
                # Should be called twice (once for each .txt file)
                assert mock_agent.learn.call_count == 2

    def test_embed_with_insight_detection(self, runner, mock_factory, mock_agent):
        """Test embedding that triggers insight detection."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Systems integration creates emergent properties.")
            temp_path = Path(f.name)

        try:
            mock_agent.learn.return_value = {
                "success": True,
                "episodes_added": 1,
                "graph_updated": True,
                "insights": [
                    {
                        "type": "emergence",
                        "content": "New pattern detected in systems integration",
                        "confidence": 0.9,
                    }
                ],
            }

            with patch.object(mock_factory, "get_agent", return_value=mock_agent):
                result = runner.invoke(app, ["embed", str(temp_path)], obj=mock_factory)

                assert result.exit_code == 0
                assert "Insights discovered" in result.stdout
                assert "emergence" in result.stdout
        finally:
            temp_path.unlink()


class TestSpikeStats:
    """Test the stats command."""

    def test_stats_display(self, runner, mock_factory, mock_agent):
        """Test displaying agent statistics."""
        with patch.object(mock_factory, "get_agent", return_value=mock_agent):
            result = runner.invoke(app, ["stats"], obj=mock_factory)

            assert result.exit_code == 0
            assert "Agent Statistics" in result.stdout
            assert "Total cycles: 10" in result.stdout
            assert "Total episodes: 100" in result.stdout
            assert "Average quality: 0.75" in result.stdout


class TestSpikeConfig:
    """Test the config command group."""

    def test_config_show(self, runner, mock_factory):
        """Test showing current configuration."""
        result = runner.invoke(app, ["config", "show"], obj=mock_factory)

        assert result.exit_code == 0
        assert "Current Configuration" in result.stdout
        assert "environment: development" in result.stdout
        assert "llm:" in result.stdout
        assert "provider: mock" in result.stdout

    def test_config_list_presets(self, runner, mock_factory):
        """Test listing available presets."""
        result = runner.invoke(app, ["config", "list-presets"], obj=mock_factory)

        assert result.exit_code == 0
        assert "Available Configuration Presets" in result.stdout
        assert "development" in result.stdout
        assert "experiment" in result.stdout
        assert "production" in result.stdout
        assert "research" in result.stdout

    def test_config_export(self, runner, mock_factory):
        """Test exporting configuration to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "config.json"

            result = runner.invoke(
                app, ["config", "export", str(output_path)], obj=mock_factory
            )

            assert result.exit_code == 0
            assert output_path.exists()

            # Verify exported content
            with open(output_path) as f:
                config_data = json.load(f)
                assert config_data["environment"] == "development"
                assert config_data["llm"]["provider"] == "mock"

    def test_config_validate(self, runner, mock_factory):
        """Test configuration validation."""
        # Create a valid config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "environment": "production",
                    "llm": {"provider": "openai", "model": "gpt-4", "temperature": 0.5},
                },
                f,
            )
            valid_path = Path(f.name)

        try:
            result = runner.invoke(
                app, ["config", "validate", str(valid_path)], obj=mock_factory
            )

            assert result.exit_code == 0
            assert "Configuration is valid" in result.stdout
        finally:
            valid_path.unlink()

        # Test invalid config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"llm": {"provider": "invalid_provider"}}, f)
            invalid_path = Path(f.name)

        try:
            result = runner.invoke(
                app, ["config", "validate", str(invalid_path)], obj=mock_factory
            )

            assert result.exit_code != 0
            assert "Validation error" in result.stdout
        finally:
            invalid_path.unlink()


class TestSpikeChat:
    """Test the interactive chat command."""

    def test_chat_single_exchange(self, runner, mock_factory, mock_agent):
        """Test a single question-answer exchange in chat mode."""
        with patch.object(mock_factory, "get_agent", return_value=mock_agent):
            # Simulate user input
            with patch("rich.prompt.Prompt.ask", side_effect=["Hello", "/exit"]):
                result = runner.invoke(app, ["chat"], obj=mock_factory)

                assert result.exit_code == 0
                assert "Interactive Chat Mode" in result.stdout
                assert mock_agent.process_question.called


class TestDependencyFactory:
    """Test the DependencyFactory."""

    def test_factory_creates_agents_with_presets(self, mock_datastore):
        """Test that factory creates agents with different presets."""
        config = ConfigPresets.development()
        factory = DependencyFactory(config, mock_datastore)

        # Mock MainAgent initialization
        with patch("insightspike.cli.spike.MainAgent") as MockMainAgent:
            mock_instance = Mock()
            mock_instance.initialize.return_value = True
            mock_instance.load_state.return_value = None
            MockMainAgent.return_value = mock_instance

            # Get development agent
            dev_agent = factory.get_agent("development")
            assert dev_agent is not None
            MockMainAgent.assert_called_once()

            # Should return cached instance
            dev_agent2 = factory.get_agent("development")
            assert dev_agent2 is dev_agent

            # Different preset should create new agent
            exp_agent = factory.get_agent("experiment")
            assert exp_agent is not dev_agent

    def test_factory_merges_base_config(self, mock_datastore):
        """Test that factory merges base config with preset."""
        base_config = InsightSpikeConfig(
            llm=LLMConfig(temperature=0.9, max_tokens=2048)
        )
        factory = DependencyFactory(base_config, mock_datastore)

        with patch("insightspike.cli.spike.MainAgent") as MockMainAgent:
            mock_instance = Mock()
            mock_instance.initialize.return_value = True
            mock_instance.load_state.return_value = None
            MockMainAgent.return_value = mock_instance

            factory.get_agent("development")

            # Check that the merged config was passed
            call_args = MockMainAgent.call_args[1]
            passed_config = call_args["config"]

            # Should have development preset values
            assert passed_config.environment == "development"
            assert passed_config.llm.provider == "mock"

            # But with base config overrides
            assert passed_config.llm.temperature == 0.9
            assert passed_config.llm.max_tokens == 2048


class TestErrorHandling:
    """Test error handling in CLI commands."""

    def test_query_handles_agent_error(self, runner, mock_factory, mock_agent):
        """Test query command handles agent errors gracefully."""
        mock_agent.process_question.side_effect = Exception("Agent error")

        with patch.object(mock_factory, "get_agent", return_value=mock_agent):
            result = runner.invoke(app, ["query", "Test question"], obj=mock_factory)

            assert result.exit_code == 1
            assert "Error:" in result.stdout

    def test_embed_handles_file_not_found(self, runner, mock_factory, mock_agent):
        """Test embed command handles missing files."""
        with patch.object(mock_factory, "get_agent", return_value=mock_agent):
            result = runner.invoke(
                app, ["embed", "/nonexistent/file.txt"], obj=mock_factory
            )

            assert result.exit_code == 1
            assert "not found" in result.stdout
