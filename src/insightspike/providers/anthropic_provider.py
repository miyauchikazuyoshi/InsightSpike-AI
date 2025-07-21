"""
Anthropic Claude LLM Provider
=============================

Real Anthropic API integration for InsightSpike.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Union

from ..config.models import LLMConfig

logger = logging.getLogger(__name__)


class AnthropicProvider:
    """Anthropic Claude provider implementation"""

    def __init__(self, config: Optional[Union[LLMConfig, Dict[str, Any]]] = None):
        """Initialize Anthropic provider"""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic library not installed. Run: pip install anthropic"
            )

        # Handle config
        if config:
            if isinstance(config, dict):
                # Dict config
                self.api_key = config.get("api_key")
                self.model = config.get("model_name", "claude-3-sonnet-20240229")
                self.temperature = config.get("temperature", 0.7)
                self.max_tokens = config.get("max_tokens", 1000)
            else:
                # Pydantic config
                self.api_key = config.api_key
                self.model = config.model or "claude-3-sonnet-20240229"
                self.temperature = getattr(config, "temperature", 0.7)
                self.max_tokens = getattr(config, "max_tokens", 1000)
        else:
            # Use environment variable
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            self.model = "claude-3-sonnet-20240229"
            self.temperature = 0.7
            self.max_tokens = 1000

        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or provide it in config"
            )

        # Initialize client
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from Claude"""
        try:
            # Merge kwargs with defaults
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            model = kwargs.get("model", self.model)

            # Call Anthropic API
            message = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract response
            return message.content[0].text

        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise

    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Async generation"""
        # Use sync client for now (can be upgraded to async client)
        return self.generate(prompt, **kwargs)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for Claude"""
        # Claude uses a similar tokenization to GPT models
        # Rough estimation: ~4 characters per token
        return len(text) // 4

    def validate_config(self) -> bool:
        """Validate Anthropic configuration"""
        try:
            # Test API key with a minimal request
            self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}],
            )
            return True
        except Exception as e:
            logger.error(f"Anthropic config validation failed: {e}")
            return False
