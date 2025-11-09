"""
Test Refactored Architecture (legacy).
Skip in cloud-safe runs when refactored_main_agent is unavailable.
"""

import pytest
pytest.importorskip(
    "insightspike.agents.refactored_main_agent",
    reason="Legacy refactor module not present in current tree.",
)
from unittest.mock import Mock, MagicMock
import numpy as np

from insightspike.config import load_config
from insightspike.agents.refactored_main_agent import RefactoredMainAgent
from insightspike.core import (
    Episode,
    CycleResult,
    ReasoningEngine,
    MemoryController,
    ResponseGenerator
)
from insightspike.di import DIContainer
from insightspike.interfaces import (
    IDataStore,
    IEmbedder,
    ILLMProvider,
    IMemoryManager,
    IMemorySearch
)


@pytest.fixture
def mock_container():
    """Create a mock DI container for testing."""
    container = DIContainer()
    
    # Mock data store
    mock_datastore = Mock(spec=IDataStore)
    mock_datastore.add_episode.return_value = None
    mock_datastore.list_episodes.return_value = []
    container.instance(IDataStore, mock_datastore)
    
    # Mock embedder
    mock_embedder = Mock(spec=IEmbedder)
    mock_embedder.encode.return_value = np.random.rand(384)
    container.instance(IEmbedder, mock_embedder)
    
    # Mock LLM provider
    mock_llm = Mock(spec=ILLMProvider)
    mock_llm.generate.return_value = {
        "text": "Test response",
        "reasoning": "Test reasoning"
    }
    container.instance(ILLMProvider, mock_llm)
    
    # Mock memory manager
    mock_memory = Mock(spec=IMemoryManager)
    mock_memory.add_episode.return_value = None
    mock_memory.search_similar.return_value = []
    mock_memory.get_statistics.return_value = {"episodes": 0}
    container.instance(IMemoryManager, mock_memory)
    
    # Mock memory search
    mock_search = Mock(spec=IMemorySearch)
    mock_search.search_with_graph.return_value = []
    container.instance(IMemorySearch, mock_search)
    
    return container


def test_refactored_agent_initialization():
    """Test RefactoredMainAgent initialization."""
    config = load_config(preset="testing")
    agent = RefactoredMainAgent(config)
    
    assert agent.config == config
    assert agent.container is not None
    assert hasattr(agent, 'reasoning_engine')
    assert hasattr(agent, 'memory_controller')
    assert hasattr(agent, 'response_generator')


def test_dependency_injection_setup(mock_container):
    """Test DI container setup."""
    config = load_config(preset="testing")
    agent = RefactoredMainAgent(config, container=mock_container)
    
    # Verify components are resolved
    assert agent.reasoning_engine is not None
    assert agent.memory_controller is not None
    assert agent.response_generator is not None


def test_process_question_basic():
    """Test basic question processing."""
    config = load_config(preset="testing")
    agent = RefactoredMainAgent(config)
    
    # Mock the reasoning engine
    mock_result = CycleResult(
        response="Test answer",
        reasoning_trace="Test reasoning",
        memory_used=[],
        spike_detected=False,
        graph_metrics={},
        reasoning_quality=0.8,
        convergence_score=0.95,
        has_spike=False
    )
    
    agent.reasoning_engine.execute_cycle = Mock(return_value=mock_result)
    agent.reasoning_engine.check_convergence = Mock(return_value=True)
    
    result = agent.process_question("Test question?")
    
    assert isinstance(result, CycleResult)
    assert "Test answer" in result.response
    assert result.reasoning_quality == 0.8


def test_add_knowledge():
    """Test adding knowledge."""
    config = load_config(preset="testing")
    agent = RefactoredMainAgent(config)
    
    # Mock memory controller
    mock_episode = Episode(
        text="Test knowledge",
        embedding=np.random.rand(384),
        c=0.5
    )
    agent.memory_controller.add_knowledge = Mock(return_value=mock_episode)
    
    agent.add_knowledge("Test knowledge")
    
    agent.memory_controller.add_knowledge.assert_called_once_with(
        "Test knowledge", None
    )


def test_memory_operations():
    """Test memory operations."""
    config = load_config(preset="testing")
    agent = RefactoredMainAgent(config)
    
    # Mock memory controller methods
    agent.memory_controller.get_memory_state = Mock(return_value={
        "episode_count": 10,
        "average_c_value": 0.7
    })
    agent.memory_controller.clear_memory = Mock()
    
    # Test get memory state
    state = agent.get_memory_state()
    assert state["episode_count"] == 10
    assert state["average_c_value"] == 0.7
    
    # Test clear memory
    agent.clear_memory(preserve_high_value=True)
    agent.memory_controller.clear_memory.assert_called_once_with(True)


def test_conversation_history():
    """Test conversation history tracking."""
    config = load_config(preset="testing")
    agent = RefactoredMainAgent(config)
    
    # Mock reasoning
    mock_result = CycleResult(
        response="Answer 1",
        reasoning_trace="",
        memory_used=[],
        spike_detected=False,
        graph_metrics={},
        reasoning_quality=0.8,
        convergence_score=0.95,
        has_spike=False
    )
    
    agent.reasoning_engine.execute_cycle = Mock(return_value=mock_result)
    agent.reasoning_engine.check_convergence = Mock(return_value=True)
    
    # Process multiple questions
    agent.process_question("Question 1")
    agent.process_question("Question 2")
    
    history = agent.get_conversation_history()
    assert len(history) == 2
    assert history[0]["question"] == "Question 1"
    assert history[1]["question"] == "Question 2"


def test_reasoning_engine_integration():
    """Test ReasoningEngine component."""
    # Create mocks
    mock_llm = Mock(spec=ILLMProvider)
    mock_search = Mock(spec=IMemorySearch)
    mock_graph_reasoner = Mock()
    mock_llm_interface = Mock()
    
    # Create reasoning engine
    engine = ReasoningEngine(
        mock_llm,
        mock_search,
        mock_graph_reasoner,
        mock_llm_interface
    )
    
    # Mock methods
    mock_graph_reasoner.analyze_documents.return_value = {
        "spike_detected": True,
        "metrics": {"ged": 0.8},
        "reasoning_quality": 0.85
    }
    
    mock_llm_interface.generate_response.return_value = {
        "text": "Generated response",
        "reasoning": "Step by step reasoning"
    }
    
    # Execute cycle
    result = engine.execute_cycle("Test question", {}, False)
    
    assert isinstance(result, CycleResult)
    assert result.spike_detected is True
    assert result.response == "Generated response"
    assert result.reasoning_quality > 0


def test_memory_controller_integration():
    """Test MemoryController component."""
    # Create mocks
    mock_memory_manager = Mock(spec=IMemoryManager)
    mock_datastore = Mock(spec=IDataStore)
    mock_embedder = Mock(spec=IEmbedder)
    mock_graph_builder = Mock()
    
    # Create memory controller
    controller = MemoryController(
        mock_memory_manager,
        mock_datastore,
        mock_embedder,
        mock_graph_builder
    )
    
    # Mock embedder
    mock_embedder.encode.return_value = np.random.rand(384)
    
    # Test add knowledge
    episode = controller.add_knowledge("Test knowledge")
    
    assert isinstance(episode, Episode)
    assert episode.text == "Test knowledge"
    mock_memory_manager.add_episode.assert_called_once()
    mock_datastore.add_episode.assert_called_once()


def test_response_generator_integration():
    """Test ResponseGenerator component."""
    mock_llm = Mock(spec=ILLMProvider)
    
    generator = ResponseGenerator(mock_llm, "detailed")
    
    # Create test cycle results
    cycle_results = [
        CycleResult(
            response="Answer 1",
            reasoning_trace="Reasoning 1",
            memory_used=[{"text": "Memory 1"}],
            spike_detected=False,
            graph_metrics={},
            reasoning_quality=0.6,
            convergence_score=0.5,
            has_spike=False
        ),
        CycleResult(
            response="Answer 2",
            reasoning_trace="Reasoning 2",
            memory_used=[{"text": "Memory 2"}],
            spike_detected=True,
            graph_metrics={},
            reasoning_quality=0.9,
            convergence_score=0.95,
            has_spike=True
        )
    ]
    
    # Generate response
    response = generator.generate_final_response(
        cycle_results,
        "Test question?",
        {}
    )
    
    assert isinstance(response, str)
    assert "Answer 2" in response  # Should select higher quality
    assert "Test question?" in response  # Should include question


def test_error_handling():
    """Test error handling in refactored agent."""
    config = load_config(preset="testing")
    agent = RefactoredMainAgent(config)
    
    # Mock reasoning engine to raise error
    agent.reasoning_engine.execute_cycle = Mock(
        side_effect=Exception("Test error")
    )
    
    # Should handle error gracefully
    result = agent.process_question("Test question?")
    
    assert isinstance(result, CycleResult)
    assert "Error" in result.response or result.reasoning_quality == 0.0


@pytest.mark.parametrize("response_style,expected_format", [
    ("concise", "simple"),
    ("detailed", "comprehensive"),
    ("academic", "formal"),
    ("conversational", "friendly")
])
def test_response_styles(response_style, expected_format):
    """Test different response formatting styles."""
    mock_llm = Mock(spec=ILLMProvider)
    generator = ResponseGenerator(mock_llm, response_style)
    
    template = generator.ResponseTemplate(
        question="What is X?",
        answer="X is Y",
        reasoning="Because of Z",
        confidence=0.85,
        sources=["Source 1", "Source 2"]
    )
    
    # Get formatter
    formatter = generator._response_templates[response_style]
    response = formatter(template)
    
    assert isinstance(response, str)
    assert "X is Y" in response
    
    if response_style == "detailed":
        assert "Question:" in response
        assert "Sources:" in response
    elif response_style == "academic":
        assert "Query:" in response
        assert "References:" in response
