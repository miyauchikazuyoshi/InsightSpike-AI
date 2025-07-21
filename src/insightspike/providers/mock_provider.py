"""
Mock LLM Provider
=================

Mock provider for testing and development.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import random
import time

from ..config.models import LLMConfig

logger = logging.getLogger(__name__)


class MockProvider:
    """Mock LLM provider for testing"""

    def __init__(self, config: Optional[Union[LLMConfig, Dict[str, Any]]] = None):
        """Initialize mock provider"""
        # Handle config
        if config:
            if isinstance(config, dict):
                # Dict config
                self.model = config.get("model_name", "mock-model")
                self.temperature = config.get("temperature", 0.7)
                self.max_tokens = config.get("max_tokens", 100)
            else:
                # Pydantic config
                self.model = config.model or "mock-model"
                self.temperature = getattr(config, "temperature", 0.7)
                self.max_tokens = getattr(config, "max_tokens", 100)
        else:
            self.model = "mock-model"
            self.temperature = 0.7
            self.max_tokens = 100

        logger.info(f"Initialized MockProvider with model: {self.model}")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate mock response"""
        # Simulate processing time
        time.sleep(0.1)

        # Generate response based on prompt
        responses = {
            "insight": "This represents a potential breakthrough in understanding the relationship between quantum mechanics and consciousness.",
            "question": "The connection between these concepts suggests a deeper underlying pattern in nature.",
            "analysis": "By examining the structural similarities, we can identify key principles that govern both domains.",
            "default": "I understand your query. Based on the available information, here's my analysis of the situation.",
        }

        # Simple keyword matching
        prompt_lower = prompt.lower()
        if "insight" in prompt_lower:
            response_type = "insight"
        elif "?" in prompt:
            response_type = "question"
        elif "analyze" in prompt_lower or "analysis" in prompt_lower:
            response_type = "analysis"
        else:
            response_type = "default"

        base_response = responses[response_type]

        # Add some variation
        variations = [
            " This pattern emerges from the underlying structure.",
            " The implications are quite significant.",
            " Further investigation reveals interesting connections.",
            " This aligns with our current understanding.",
        ]

        response = base_response + random.choice(variations)

        # Respect max_tokens (rough approximation)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        words = response.split()
        if len(words) > max_tokens // 4:  # Rough token estimation
            words = words[: max_tokens // 4]
            response = " ".join(words) + "..."

        return response

    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Async generation (just calls sync version)"""
        return self.generate(prompt, **kwargs)

    def generate_embeddings(self, text: str) -> List[float]:
        """Generate mock embeddings"""
        # Simple hash-based embedding
        import hashlib
        import numpy as np

        # Hash the text
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()

        # Convert to floats
        np.random.seed(int.from_bytes(hash_bytes[:4], "big"))
        embedding = np.random.randn(384).tolist()

        return embedding

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count"""
        # Simple estimation: ~4 characters per token
        return len(text) // 4

    def validate_config(self) -> bool:
        """Mock provider is always valid"""
        return True
