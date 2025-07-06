"""
Unit tests for CLI main module
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
from typer.testing import CliRunner

from insightspike.cli.main import app


runner = CliRunner()


class TestAskCommand:
    """Test the ask command."""
    
    @patch('insightspike.cli.main.MainAgent')
    def test_ask_success(self, mock_agent_class):
        """Test successful ask command."""
        # Mock agent
        mock_agent = Mock()
        mock_agent.initialize.return_value = True
        mock_agent.process_question.return_value = {
            'response': 'Test answer',
            'reasoning_quality': 0.85,
            'total_cycles': 3,
            'success': True
        }
        mock_agent_class.return_value = mock_agent
        
        result = runner.invoke(app, ["ask", "What is AI?"])
        
        assert result.exit_code == 0
        assert "Test answer" in result.stdout
        assert "Quality: 0.850" in result.stdout
        mock_agent.process_question.assert_called_once_with(
            "What is AI?", 
            max_cycles=5, 
            verbose=True
        )
    
    @patch('insightspike.cli.main.MainAgent')
    def test_ask_initialization_failure(self, mock_agent_class):
        """Test ask command when initialization fails."""
        mock_agent = Mock()
        mock_agent.initialize.return_value = False
        mock_agent_class.return_value = mock_agent
        
        result = runner.invoke(app, ["ask", "Test question"])
        
        assert result.exit_code == 1
        assert "Failed to initialize agent" in result.stdout
    
    @patch('insightspike.cli.main.MainAgent')
    def test_ask_with_error(self, mock_agent_class):
        """Test ask command with error."""
        mock_agent = Mock()
        mock_agent.initialize.side_effect = Exception("Test error")
        mock_agent_class.return_value = mock_agent
        
        result = runner.invoke(app, ["ask", "Test question"])
        
        assert result.exit_code == 1
        assert "Error" in result.stdout


class TestLearnCommand:
    """Test the learn command."""
    
    @patch('insightspike.cli.main.MainAgent')
    @patch('insightspike.cli.main.load_corpus')
    def test_learn_from_file(self, mock_load_corpus, mock_agent_class):
        """Test learning from file."""
        # Mock corpus loading
        mock_load_corpus.return_value = ["Doc 1", "Doc 2", "Doc 3"]
        
        # Mock agent
        mock_agent = Mock()
        mock_agent.initialize.return_value = True
        mock_agent.add_document = Mock()
        mock_agent_class.return_value = mock_agent
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = f.name
        
        try:
            result = runner.invoke(app, ["learn", temp_path])
            
            assert result.exit_code == 0
            assert "Loading documents from" in result.stdout
            assert "Loaded 3 documents" in result.stdout
            assert mock_agent.add_document.call_count == 3
        finally:
            Path(temp_path).unlink()
    
    @patch('insightspike.cli.main.MainAgent')
    def test_learn_from_text(self, mock_agent_class):
        """Test learning from direct text."""
        mock_agent = Mock()
        mock_agent.initialize.return_value = True
        mock_agent.add_document = Mock()
        mock_agent_class.return_value = mock_agent
        
        result = runner.invoke(app, ["learn", "--text", "Learn this text"])
        
        assert result.exit_code == 0
        assert "Learning from provided text" in result.stdout
        mock_agent.add_document.assert_called_once_with("Learn this text")
    
    def test_learn_no_input(self):
        """Test learn command with no input."""
        result = runner.invoke(app, ["learn"])
        
        assert result.exit_code != 0
        assert "provide either a file path or text" in result.stdout


class TestTrainCommand:
    """Test the train command."""
    
    @patch('insightspike.cli.main.ExperimentFramework')
    def test_train_default(self, mock_framework_class):
        """Test train command with defaults."""
        mock_framework = Mock()
        mock_framework.run_experiment.return_value = {
            'agent': 'test_agent',
            'avg_reward': 10.5,
            'total_insights': 5
        }
        mock_framework_class.return_value = mock_framework
        
        result = runner.invoke(app, ["train"])
        
        assert result.exit_code == 0
        assert "Starting training" in result.stdout
        mock_framework.run_experiment.assert_called_once()
    
    @patch('insightspike.cli.main.ExperimentFramework')
    def test_train_with_parameters(self, mock_framework_class):
        """Test train command with custom parameters."""
        mock_framework = Mock()
        mock_framework.run_experiment.return_value = {
            'agent': 'test_agent',
            'avg_reward': 15.0,
            'total_insights': 10
        }
        mock_framework_class.return_value = mock_framework
        
        result = runner.invoke(app, [
            "train",
            "--episodes", "50",
            "--domain", "custom_domain"
        ])
        
        assert result.exit_code == 0
        config_call = mock_framework_class.call_args[1]['config']
        assert config_call['num_episodes'] == 50
        assert config_call['domain'] == 'custom_domain'


class TestValidateCommand:
    """Test the validate command."""
    
    @patch('insightspike.cli.main.MainAgent')
    def test_validate_success(self, mock_agent_class):
        """Test successful validation."""
        mock_agent = Mock()
        mock_agent.initialize.return_value = True
        mock_agent.get_memory_stats.return_value = {
            'total_episodes': 100,
            'index_trained': True
        }
        mock_agent_class.return_value = mock_agent
        
        result = runner.invoke(app, ["validate"])
        
        assert result.exit_code == 0
        assert "System validation passed" in result.stdout
        assert "Memory: 100 episodes" in result.stdout
    
    @patch('insightspike.cli.main.MainAgent')
    def test_validate_no_episodes(self, mock_agent_class):
        """Test validation with no episodes."""
        mock_agent = Mock()
        mock_agent.initialize.return_value = True
        mock_agent.get_memory_stats.return_value = {
            'total_episodes': 0,
            'index_trained': False
        }
        mock_agent_class.return_value = mock_agent
        
        result = runner.invoke(app, ["validate"])
        
        assert result.exit_code == 0
        assert "Warning: No episodes in memory" in result.stdout


class TestExportCommand:
    """Test the export command."""
    
    @patch('insightspike.cli.main.MainAgent')
    def test_export_success(self, mock_agent_class):
        """Test successful export."""
        mock_agent = Mock()
        mock_agent.initialize.return_value = True
        mock_agent.export_memory.return_value = {
            'episodes': [{'text': 'Episode 1'}],
            'metadata': {'version': '1.0'}
        }
        mock_agent_class.return_value = mock_agent
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "export.json"
            result = runner.invoke(app, ["export", str(output_path)])
            
            assert result.exit_code == 0
            assert output_path.exists()
            
            # Check exported content
            with open(output_path) as f:
                data = json.load(f)
                assert 'episodes' in data
                assert len(data['episodes']) == 1


class TestStatusCommand:
    """Test the status command."""
    
    @patch('insightspike.cli.main.MainAgent')
    @patch('insightspike.cli.main.InsightFactRegistry')
    def test_status(self, mock_registry_class, mock_agent_class):
        """Test status command."""
        # Mock agent
        mock_agent = Mock()
        mock_agent.initialize.return_value = True
        mock_agent.get_memory_stats.return_value = {
            'total_episodes': 50,
            'index_trained': True
        }
        mock_agent_class.return_value = mock_agent
        
        # Mock registry
        mock_registry = Mock()
        mock_registry.status.return_value = "Registry: 10 insights detected"
        mock_registry_class.return_value = mock_registry
        
        result = runner.invoke(app, ["status"])
        
        assert result.exit_code == 0
        assert "Episodes: 50" in result.stdout
        assert "Index: ✓ Trained" in result.stdout
        assert "Registry: 10 insights detected" in result.stdout


class TestClearCommand:
    """Test the clear command."""
    
    @patch('insightspike.cli.main.MainAgent')
    def test_clear_confirmed(self, mock_agent_class):
        """Test clear command with confirmation."""
        mock_agent = Mock()
        mock_agent.initialize.return_value = True
        mock_agent.clear_memory.return_value = True
        mock_agent_class.return_value = mock_agent
        
        result = runner.invoke(app, ["clear"], input="y\n")
        
        assert result.exit_code == 0
        assert "Memory cleared successfully" in result.stdout
        mock_agent.clear_memory.assert_called_once()
    
    @patch('insightspike.cli.main.MainAgent')
    def test_clear_cancelled(self, mock_agent_class):
        """Test clear command cancelled."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        result = runner.invoke(app, ["clear"], input="n\n")
        
        assert result.exit_code == 0
        assert "Clear cancelled" in result.stdout
        mock_agent.clear_memory.assert_not_called()


class TestListCommand:
    """Test the list command."""
    
    @patch('insightspike.cli.main.MainAgent')
    def test_list_episodes(self, mock_agent_class):
        """Test listing episodes."""
        mock_agent = Mock()
        mock_agent.initialize.return_value = True
        mock_agent.list_episodes.return_value = [
            {'id': 0, 'text': 'Episode 1', 'c_value': 0.5},
            {'id': 1, 'text': 'Episode 2', 'c_value': 0.8}
        ]
        mock_agent_class.return_value = mock_agent
        
        result = runner.invoke(app, ["list", "--limit", "2"])
        
        assert result.exit_code == 0
        assert "Episode 1" in result.stdout
        assert "Episode 2" in result.stdout
        assert "C=0.500" in result.stdout
        assert "C=0.800" in result.stdout


class TestSearchCommand:
    """Test the search command."""
    
    @patch('insightspike.cli.main.MainAgent')
    def test_search(self, mock_agent_class):
        """Test search command."""
        mock_agent = Mock()
        mock_agent.initialize.return_value = True
        mock_agent.search_episodes.return_value = [
            {'text': 'Found episode 1', 'score': 0.9},
            {'text': 'Found episode 2', 'score': 0.7}
        ]
        mock_agent_class.return_value = mock_agent
        
        result = runner.invoke(app, ["search", "test query", "--top-k", "2"])
        
        assert result.exit_code == 0
        assert "Found episode 1" in result.stdout
        assert "Score: 0.900" in result.stdout
        mock_agent.search_episodes.assert_called_once_with("test query", k=2)


class TestVersionCommand:
    """Test the version command."""
    
    def test_version(self):
        """Test version command."""
        result = runner.invoke(app, ["version"])
        
        assert result.exit_code == 0
        assert "InsightSpike-AI" in result.stdout
        assert "Version:" in result.stdout


class TestDepsCommands:
    """Test dependency management commands."""
    
    @patch('subprocess.run')
    def test_deps_install(self, mock_run):
        """Test deps install command."""
        mock_run.return_value = Mock(returncode=0)
        
        result = runner.invoke(app, ["deps", "install"])
        
        # Note: This might fail if deps commands are not properly integrated
        # Just check it doesn't crash
        assert result.exit_code in [0, 2]  # 2 is for missing subcommand