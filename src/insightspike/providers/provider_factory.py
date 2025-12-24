"""
LLM Provider Factory
====================

Factory for creating appropriate LLM providers based on configuration.
"""

import logging
from typing import Any, Dict, Optional, Union

from ..config.models import LLMConfig
from .mock_provider import MockProvider

# Import providers conditionally
try:
    from .openai_provider import OpenAIProvider
except ImportError:
    OpenAIProvider = None

try:
    from .anthropic_provider import AnthropicProvider
except ImportError:
    AnthropicProvider = None

try:
    from .local import LocalProvider
except ImportError:
    LocalProvider = None

try:
    from .simple_local import SimpleLocalProvider
except ImportError:
    SimpleLocalProvider = None

try:
    from .distilgpt2_provider import DistilGPT2Provider
except ImportError:
    DistilGPT2Provider = None

logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory for creating LLM providers"""

    # Registry of available providers
    PROVIDERS = {
        "mock": MockProvider,
    }

    # Add providers if available
    if OpenAIProvider:
        PROVIDERS["openai"] = OpenAIProvider
        PROVIDERS["gpt"] = OpenAIProvider  # Alias
        PROVIDERS["ollama"] = OpenAIProvider  # Alias for local LLM (OpenAI compatible)

    if AnthropicProvider:
        PROVIDERS["anthropic"] = AnthropicProvider
        PROVIDERS["claude"] = AnthropicProvider  # Alias

    if LocalProvider:
        PROVIDERS["local"] = LocalProvider
    elif SimpleLocalProvider:
        # Fallback to simple local if main local provider fails
        PROVIDERS["local"] = SimpleLocalProvider
    elif DistilGPT2Provider:
        # Fallback to DistilGPT2 provider
        PROVIDERS["local"] = DistilGPT2Provider

    if DistilGPT2Provider:
        PROVIDERS["distilgpt2"] = DistilGPT2Provider

    @classmethod
    def create(
        cls,
        provider_name: str,
        config: Optional[Union[LLMConfig, Dict[str, Any]]] = None,
    ) -> Any:
        """
        Create an LLM provider instance.

        Args:
            provider_name: Name of the provider (mock, openai, anthropic)
            config: LLM configuration

        Returns:
            Provider instance

        Raises:
            ValueError: If provider not found
        """
        provider_name = provider_name.lower()

        if provider_name not in cls.PROVIDERS:
            available = ", ".join(cls.PROVIDERS.keys())
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available providers: {available}"
            )

        provider_class = cls.PROVIDERS[provider_name]

        try:
            provider = provider_class(config)
            logger.info(f"Created {provider_name} provider")
            return provider

        except ImportError as e:
            logger.error(f"Failed to import {provider_name} dependencies: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to create {provider_name} provider: {e}")
            raise

    @classmethod
    def create_from_config(cls, config: Union[LLMConfig, Dict[str, Any]]) -> Any:
        """
        Create provider from config object.

        Args:
            config: LLM configuration with 'provider' field

        Returns:
            Provider instance
        """
        if hasattr(config, "provider"):
            provider_name = config.provider
        else:
            provider_name = config.get("provider", "mock")

        return cls.create(provider_name, config)

    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """
        Register a custom provider.

        Args:
            name: Provider name
            provider_class: Provider class
        """
        cls.PROVIDERS[name.lower()] = provider_class
        logger.info(f"Registered provider: {name}")
