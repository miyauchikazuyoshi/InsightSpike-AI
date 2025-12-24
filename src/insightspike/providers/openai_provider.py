"""
OpenAI LLM Provider
===================

Real OpenAI API integration for InsightSpike.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

from ..config.models import LLMConfig

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """OpenAI LLM provider implementation"""

    def __init__(self, config: Optional[Union[LLMConfig, Dict[str, Any]]] = None):
        """Initialize OpenAI provider"""
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")

        # Handle config
        if config:
            if isinstance(config, dict):
                # Dict config
                self.api_key = config.get("api_key")
                self.base_url = config.get("api_base")
                self.model = config.get("model_name", "gpt-3.5-turbo")
                self.temperature = config.get("temperature", 0.7)
                self.max_tokens = config.get("max_tokens", 1000)
            else:
                # Pydantic config
                self.api_key = config.api_key
                self.base_url = config.api_base
                self.model = config.model or "gpt-3.5-turbo"
                self.temperature = getattr(config, "temperature", 0.7)
                self.max_tokens = getattr(config, "max_tokens", 1000)
        else:
            # Use environment variable
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.base_url = os.getenv("OPENAI_API_BASE")
            self.model = "gpt-3.5-turbo"
            self.temperature = 0.7
            self.max_tokens = 1000

        if not self.api_key:
            # For local providers (via OpenAI protocol), API key might be dummy
            if not self.base_url:
                 raise ValueError(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                    "or provide it in config"
                 )
            # If base_url is set (e.g. local), we can tolerate missing key if strictly needed
            if not self.api_key:
                self.api_key = "dummy"

        # Initialize client
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from OpenAI"""
        try:
            # Merge kwargs with defaults
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            model = kwargs.get("model", self.model)

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Extract response
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Async generation (OpenAI client handles this internally)"""
        # OpenAI's client is already async-compatible
        return self.generate(prompt, **kwargs)

    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002", input=text
            )
            return response.data[0].embedding

        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}")
            raise

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count"""
        # Rough estimation: ~4 characters per token
        return len(text) // 4

    def validate_config(self) -> bool:
        """Validate OpenAI configuration"""
        try:
            # Test API key with a minimal request
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"OpenAI config validation failed: {e}")
            return False
