"""Quick start helpers for InsightSpike-AI."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

from pydantic import ValidationError

from .config import load_config
from .implementations.agents.main_agent import MainAgent

logger = logging.getLogger(__name__)


def create_agent(provider: str = "mock", **kwargs) -> MainAgent:
    """Create a ready-to-use InsightSpike agent with minimal configuration.
    
    Args:
        provider: LLM provider to use ('mock', 'openai', 'local', etc.)
        **kwargs: Additional configuration options. Nested fields can be
            expressed with double underscores (e.g. ``llm__temperature=0.3`` or
            ``processing__max_cycles=4``). A ``model`` keyword is treated as a
            shortcut for ``llm__model``. To select a different preset before
            overrides are applied, pass ``preset="cloud"`` (or other preset
            names) as part of ``kwargs``.
        
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

    preset_override = kwargs.pop("preset", None)
    preset = preset_override or preset_map.get(provider, "experiment")

    overrides, model_overridden = _prepare_config_overrides(provider, kwargs)
    config = load_config(preset=preset)
    config = _apply_config_overrides(config, overrides)

    # For local provider, use smaller model by default on CPU / no CUDA
    if config.llm.provider == "local" and not model_overridden:
        if not _has_cuda_support():
            config.llm.device = "cpu"
            config.llm.model = "google/flan-t5-small"  # 77MB vs 1.1GB
            logger.info(
                "Torch/CUDA not available; using CPU-friendly model: flan-t5-small"
            )

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


def _prepare_config_overrides(
    provider: str,
    extra_kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool]:
    """Convert keyword arguments into a nested override tree for the config."""
    overrides: Dict[str, Any] = {}
    model_overridden = False

    def assign(path: Tuple[str, ...], value: Any) -> None:
        nonlocal model_overridden
        if not path or any(part == "" for part in path):
            raise ValueError(f"Invalid configuration override key: {path}")
        cursor = overrides
        for part in path[:-1]:
            next_node = cursor.setdefault(part, {})
            if not isinstance(next_node, dict):
                raise ValueError(
                    f"Cannot override nested path {'__'.join(path)}; {part} already set"
                )
            cursor = next_node
        cursor[path[-1]] = value
        if path == ("llm", "model"):
            model_overridden = True

    for key, value in extra_kwargs.items():
        if key == "model":
            path = ("llm", "model")
        elif "__" in key:
            path = tuple(part.strip() for part in key.split("__") if part.strip())
        else:
            path = (key,)
        assign(path, value)

    # Explicit provider argument takes precedence.
    assign(("llm", "provider"), provider)
    return overrides, model_overridden


def _apply_config_overrides(config, overrides: Dict[str, Any]):
    """Apply nested overrides onto the config and return a new instance."""
    if not overrides:
        return config

    config_cls = config.__class__
    base = _config_to_dict(config)
    merged = _deep_merge_dicts(base, overrides)
    try:
        return config_cls(**merged)
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration override: {exc}") from exc


def _config_to_dict(config) -> Dict[str, Any]:
    if hasattr(config, "model_dump"):
        return config.model_dump()
    return config.dict()


def _deep_merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = deepcopy(base)
    for key, value in overrides.items():
        if (
            isinstance(value, dict)
            and isinstance(result.get(key), dict)
        ):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _has_cuda_support() -> bool:
    try:
        import torch  # type: ignore
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# Convenience imports
__all__ = ['create_agent', 'quick_demo']
