"""Quick start helpers for InsightSpike-AI."""

import logging
from typing import Optional, Dict, Any

from .config import load_config
from .implementations.agents.main_agent import MainAgent

logger = logging.getLogger(__name__)


def create_agent(provider: str = "mock", **kwargs) -> MainAgent:
    """Create a ready-to-use InsightSpike agent with minimal configuration.
    
    Args:
        provider: LLM provider to use ('mock', 'openai', 'local', etc.)
        **kwargs: Additional configuration options
        
    Returns:
        Initialized MainAgent ready for use
        
    Example:
        ```python
        from insightspike import create_agent
        
        # Simple usage
        agent = create_agent()
        result = agent.process_question("What is the meaning of life?")
        print(result.response)
        
        # With OpenAI
        agent = create_agent(provider="openai")  # Requires OPENAI_API_KEY env var
        
        # With custom model
        agent = create_agent(provider="local", model="google/flan-t5-small")
        ```
    """
    # Map provider to preset names
    preset_map = {
        "mock": "experiment",  # Use experiment preset for mock
        "openai": "experiment",  # No specific openai preset yet
        "anthropic": "experiment",  # No specific anthropic preset yet
        "local": "experiment",
        "clean": "experiment"
    }
    
    preset = preset_map.get(provider, "experiment")  # Default to experiment preset
    
    # Load config with preset
    config = load_config(preset=preset)
    
    # Override provider and model if specified
    if provider:
        config.llm.provider = provider
    
    if "model" in kwargs:
        config.llm.model_name = kwargs["model"]
        
    # For local provider, use smaller model by default on CPU
    if provider == "local" and "model" not in kwargs:
        import torch
        if not torch.cuda.is_available():
            config.llm.model_name = "google/flan-t5-small"  # 77MB vs 1.1GB
            logger.info("CPU detected, using smaller model: flan-t5-small")
    
    # Create and initialize agent
    agent = MainAgent(config)
    
    # Check if agent needs initialization (handle both old and new style)
    if hasattr(agent, 'initialized') and not agent.initialized:
        success = agent.initialize()
        if not success:
            logger.warning("Agent initialization had issues, but may still work")
    elif hasattr(agent, '_initialized') and not agent._initialized:
        # Some agents use _initialized instead
        logger.info("Agent auto-initialized")
    
    return agent


def quick_demo():
    """Run a quick demonstration of InsightSpike capabilities."""
    print("=== InsightSpike Quick Demo ===\n")
    
    # Create agent
    print("Creating agent...")
    agent = create_agent()
    
    # Add some knowledge
    print("Adding knowledge...")
    knowledge_items = [
        "The Earth orbits around the Sun.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Photosynthesis converts light energy into chemical energy.",
    ]
    
    for item in knowledge_items:
        agent.add_knowledge(item)
        print(f"  ✓ {item}")
    
    # Ask questions
    print("\nAsking questions...")
    questions = [
        "Why does water boil?",
        "How do plants get energy?",
        "What moves around what in our solar system?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        result = agent.process_question(question)
        
        if hasattr(result, 'response'):
            print(f"A: {result.response}")
            if hasattr(result, 'has_spike') and result.has_spike:
                print("  💡 Insight detected!")
        else:
            print(f"A: {result.get('response', 'No response')}")
    
    print("\n=== Demo Complete ===")


# Convenience imports
__all__ = ['create_agent', 'quick_demo']