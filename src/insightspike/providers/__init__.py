"""LLM Provider implementations."""

from .mock_provider import MockProvider
from .provider_factory import ProviderFactory
from ..implementations.layers.layer4_llm_interface import LLMProviderRegistry  # added for DI compatibility

# Import actual providers only if available
try:
    from .openai_provider import OpenAIProvider

    __all__ = ["MockProvider", "OpenAIProvider", "ProviderFactory", "LLMProviderRegistry"]
except ImportError:
    __all__ = ["MockProvider", "ProviderFactory", "LLMProviderRegistry"]

try:
    from .anthropic_provider import AnthropicProvider

    if "AnthropicProvider" not in __all__:
        __all__.append("AnthropicProvider")
except ImportError:
    pass

try:
    from .local import LocalProvider

    if "LocalProvider" not in __all__:
        __all__.append("LocalProvider")
except ImportError:
    pass

try:
    from .distilgpt2_provider import DistilGPT2Provider

    if "DistilGPT2Provider" not in __all__:
        __all__.append("DistilGPT2Provider")
except ImportError:
    pass
