"""
Simple Local Provider for DistilGPT2
====================================

A simplified local provider that uses cached distilgpt2 model.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import time
import random

from ..config.models import LLMConfig

logger = logging.getLogger(__name__)

# Global cache for model
_model_cache = {}


class SimpleLocalProvider:
    """Simple local provider using DistilGPT2"""

    def __init__(self, config: Optional[Union[LLMConfig, Dict[str, Any]]] = None):
        """Initialize simple local provider"""
        # Handle config
        if config:
            if isinstance(config, dict):
                self.model_name = config.get("model", "distilgpt2")
                self.temperature = config.get("temperature", 0.7)
                self.max_tokens = config.get("max_tokens", 200)
            else:
                self.model_name = config.model or "distilgpt2"
                self.temperature = getattr(config, "temperature", 0.7)
                self.max_tokens = getattr(config, "max_tokens", 200)
        else:
            self.model_name = "distilgpt2"
            self.temperature = 0.7
            self.max_tokens = 200

        logger.info(f"Initialized SimpleLocalProvider with model: {self.model_name}")

        # Lazy loading flag
        self._initialized = False
        self._pipeline = None

    def _ensure_initialized(self):
        """Ensure the model is loaded"""
        if not self._initialized:
            logger.info(f"Loading {self.model_name} on first use...")

            # Check cache first
            if self.model_name in _model_cache:
                self._pipeline = _model_cache[self.model_name]
                logger.info(f"Using cached {self.model_name}")
            else:
                # Load model
                from transformers import pipeline

                self._pipeline = pipeline(
                    "text-generation", model=self.model_name, device=-1  # CPU
                )
                _model_cache[self.model_name] = self._pipeline
                logger.info(f"Loaded and cached {self.model_name}")

            self._initialized = True

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using DistilGPT2"""
        self._ensure_initialized()

        # Simulate some processing
        time.sleep(0.05)

        try:
            # Generate with pipeline
            result = self._pipeline(
                prompt,
                max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self._pipeline.tokenizer.eos_token_id,
            )

            # Extract text
            generated = result[0]["generated_text"]

            # Remove prompt from output
            if generated.startswith(prompt):
                generated = generated[len(prompt) :].strip()

            # Limit length and clean up
            if len(generated) > 500:
                generated = generated[:500] + "..."

            return (
                generated
                or "I understand your question about the relationship between these concepts."
            )

        except Exception as e:
            logger.error(f"Generation error: {e}")
            # Fallback response
            return "The relationship between these concepts represents an interesting area of study."

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate for multiple prompts"""
        return [self.generate(p, **kwargs) for p in prompts]

    def embed(self, text: str) -> List[float]:
        """Embeddings not supported"""
        return []

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embeddings not supported"""
        return [[] for _ in texts]

    def count_tokens(self, text: str) -> int:
        """Estimate token count"""
        # Rough estimate: 1 token per 4 characters
        return len(text) // 4

    def get_model_info(self) -> Dict[str, Any]:
        """Get model info"""
        return {
            "provider": "simple_local",
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
