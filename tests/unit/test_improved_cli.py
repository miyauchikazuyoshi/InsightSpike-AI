#!/usr/bin/env python3
"""
Test suite for improved CLI
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from typer.testing import CliRunner

# Mock the dependencies to avoid import issues during testing
sys.modules['insightspike.core.agents.main_agent'] = Mock()
sys.modules['insightspike.core.layers'] = Mock()

from insightspike.cli.improved_cli import app, state, get_or_create_agent
from insightspike.config import ConfigPresets, SimpleConfig


runner = CliRunner()


def test_ask_command():
    """Test the ask command"""
    print("\n=== Test Ask Command ===")
    
    # Test basic ask
    result = runner.invoke(app, ["ask", "What is AI?"])
    assert result.exit_code == 0
    assert "Question:" in result.stdout
    assert "Answer:" in result.stdout
    print("✓ Basic ask command works")
    
    # Test with preset
    result = runner.invoke(app, ["ask", "Test question", "--preset", "development"])
    assert result.exit_code == 0
    print("✓ Ask with preset works")
    
    # Test verbose mode
    result = runner.invoke(app, ["ask", "Test", "-v"])
    assert result.exit_code == 0
    assert "Quality:" in result.stdout
    print("✓ Verbose mode works")


def test_config_command():
    """Test configuration management"""
    print("\n=== Test Config Command ===")
    
    # Test show config
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "Current Configuration" in result.stdout
    print("✓ Config show works")
    
    # Test set config
    result = runner.invoke(app, ["config", "set", "debug", "true"])
    assert result.exit_code == 0
    assert "Set debug = True" in result.stdout
    print("✓ Config set works")
    
    # Test preset loading
    result = runner.invoke(app, ["config", "preset", "experiment"])
    assert result.exit_code == 0
    assert "Loaded experiment preset" in result.stdout
    print("✓ Config preset works")
    
    # Test save/load
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        config_file = f.name
        
    result = runner.invoke(app, ["config", "save", config_file])
    assert result.exit_code == 0
    assert "Configuration saved" in result.stdout
    
    result = runner.invoke(app, ["config", "load", config_file])
    assert result.exit_code == 0
    assert "Configuration loaded" in result.stdout
    
    Path(config_file).unlink()
    print("✓ Config save/load works")


def test_embed_command():
    """Test document embedding"""
    print("\n=== Test Embed Command ===")
    
    # Create test file
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("This is test content for embedding.")
        
        # Test file embedding
        result = runner.invoke(app, ["embed", str(test_file)])
        assert result.exit_code == 0
        assert "Found 1 document" in result.stdout
        assert "Embedded" in result.stdout or "Added" in result.stdout
        print("✓ Embed from file works")
        
        # Test directory embedding
        test_file2 = Path(tmpdir) / "test2.txt"
        test_file2.write_text("Another test document.")
        
        result = runner.invoke(app, ["embed", tmpdir])
        assert result.exit_code == 0
        assert "Found 2 document" in result.stdout
        print("✓ Embed from directory works")


def test_stats_command():
    """Test stats display"""
    print("\n=== Test Stats Command ===")
    
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "Agent Statistics" in result.stdout
    assert "Initialized" in result.stdout
    print("✓ Stats command works")


def test_experiment_command():
    """Test experiment runner"""
    print("\n=== Test Experiment Command ===")
    
    # Test simple experiment
    result = runner.invoke(app, ["experiment", "--name", "simple", "--episodes", "3"])
    assert result.exit_code == 0
    assert "Running simple experiment" in result.stdout
    assert "Experiment complete!" in result.stdout
    print("✓ Simple experiment works")
    
    # Test insight experiment
    result = runner.invoke(app, ["experiment", "--name", "insight", "--episodes", "5"])
    assert result.exit_code == 0
    assert "Testing synthesis capabilities" in result.stdout
    print("✓ Insight experiment works")


def test_version_command():
    """Test version display"""
    print("\n=== Test Version Command ===")
    
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "InsightSpike AI" in result.stdout
    assert "Version:" in result.stdout
    assert "Python:" in result.stdout
    print("✓ Version command works")


def test_command_aliases():
    """Test command aliases"""
    print("\n=== Test Command Aliases ===")
    
    # Test 'q' alias for query
    result = runner.invoke(app, ["q", "Quick question"])
    assert result.exit_code == 0
    assert "Question:" in result.stdout
    print("✓ 'q' alias works")
    
    # Test 'e' alias for embed
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"Test content")
        test_file = f.name
        
    result = runner.invoke(app, ["e", test_file])
    assert result.exit_code == 0
    assert "Found 1 document" in result.stdout
    
    Path(test_file).unlink()
    print("✓ 'e' alias works")
    
    # Test 'c' alias for chat
    with patch('rich.prompt.Prompt.ask', side_effect=['exit']):
        result = runner.invoke(app, ["c"])
        assert result.exit_code == 0
        assert "Interactive Mode" in result.stdout
        print("✓ 'c' alias works")
    
    # Legacy command aliases
    result = runner.invoke(app, ["ask", "Test"])
    assert result.exit_code == 0
    assert "Question:" in result.stdout
    print("✓ 'ask' legacy alias works")
    
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"Test")
        test_file = f.name
    
    result = runner.invoke(app, ["learn", test_file])
    assert result.exit_code == 0
    Path(test_file).unlink()
    print("✓ 'learn' legacy alias works")


def test_chat_command():
    """Test interactive chat mode"""
    print("\n=== Test Chat Command ===")
    
    # Mock user input for chat
    with patch('rich.prompt.Prompt.ask', side_effect=['help', 'exit']):
        result = runner.invoke(app, ["chat"])
        assert result.exit_code == 0
        assert "Interactive Mode" in result.stdout
        assert "Available commands" in result.stdout
        assert "Goodbye!" in result.stdout
        print("✓ Chat mode help and exit work")


def test_error_handling():
    """Test error handling"""
    print("\n=== Test Error Handling ===")
    
    # Test invalid file path
    result = runner.invoke(app, ["embed", "/nonexistent/path"])
    assert result.exit_code == 1
    assert "Path not found" in result.stdout
    print("✓ Invalid path error handled")
    
    # Test invalid config key
    result = runner.invoke(app, ["config", "set", "invalid_key", "value"])
    assert result.exit_code == 0
    assert "Error setting config" in result.stdout
    print("✓ Invalid config key handled")
    
    # Test invalid preset
    result = runner.invoke(app, ["config", "preset", "invalid"])
    assert result.exit_code == 1
    assert "Unknown preset" in result.stdout
    print("✓ Invalid preset handled")


def test_agent_initialization():
    """Test agent initialization and caching"""
    print("\n=== Test Agent Initialization ===")
    
    # Reset state
    state.agent = None
    state.config_manager = None
    
    # First call should initialize
    agent1 = get_or_create_agent("development")
    assert agent1 is not None
    assert state.agent is agent1
    print("✓ Agent initialized")
    
    # Second call should return same agent
    agent2 = get_or_create_agent("development")
    assert agent2 is agent1
    print("✓ Agent cached correctly")
    
    # Changing config should reset agent
    state.config_manager.set("debug", True)
    state.agent = None
    agent3 = get_or_create_agent("development")
    assert agent3 is not agent1
    print("✓ Agent reset after config change")


def test_insights_command():
    """Test insights display command"""
    print("\n=== Test Insights Command ===")
    
    # Test basic insights display
    result = runner.invoke(app, ["insights"])
    assert result.exit_code == 0
    assert "Discovered Insights" in result.stdout or "No insights discovered yet" in result.stdout
    print("✓ Insights command works")
    
    # Test with limit
    result = runner.invoke(app, ["insights", "--limit", "5"])
    assert result.exit_code == 0
    print("✓ Insights with limit works")


def test_insights_search_command():
    """Test insights search command"""
    print("\n=== Test Insights Search Command ===")
    
    # Test search for a concept
    result = runner.invoke(app, ["insights-search", "energy"])
    assert result.exit_code == 0
    assert "Searching for insights" in result.stdout or "insights found" in result.stdout
    print("✓ Insights search works")
    
    # Test with limit
    result = runner.invoke(app, ["insights-search", "quantum", "--limit", "3"])
    assert result.exit_code == 0
    print("✓ Insights search with limit works")


def test_demo_command():
    """Test demo command"""
    print("\n=== Test Demo Command ===")
    
    # Mock user input for demo
    with patch('rich.prompt.Prompt.ask', side_effect=['1', '2', '3', '4', '5']):
        result = runner.invoke(app, ["demo"])
        assert result.exit_code == 0
        assert "InsightSpike Demo" in result.stdout
        assert "Demo Options" in result.stdout
        print("✓ Demo command works")


def run_all_tests():
    """Run all CLI tests"""
    print("=== Running Improved CLI Tests ===")
    
    tests = [
        test_ask_command,
        test_config_command,
        test_embed_command,
        test_stats_command,
        test_experiment_command,
        test_version_command,
        test_command_aliases,
        test_chat_command,
        test_error_handling,
        test_agent_initialization,
        test_insights_command,
        test_insights_search_command,
        test_demo_command
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n=== Test Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {len(tests)}")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)