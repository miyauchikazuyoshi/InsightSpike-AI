"""
Local LLM Provider
==================

Provides support for locally hosted language models using Hugging Face transformers.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import torch

from ..config.models import LLMConfig

logger = logging.getLogger(__name__)


class LocalProvider:
    """Provider for local language models using Hugging Face transformers."""

    def __init__(self, config: Optional[Union[LLMConfig, Dict[str, Any]]] = None):
        """Initialize local provider with config."""
        # Handle config
        if config:
            if isinstance(config, dict):
                # Dict config
                self.model_name = config.get("model", "distilgpt2")
                self.temperature = config.get("temperature", 0.7)
                self.max_tokens = config.get("max_tokens", 256)
                self.device = config.get("device", "cpu")
                self.load_in_8bit = config.get("load_in_8bit", False)
                self.top_p = config.get("top_p", 0.9)
            else:
                # Pydantic config
                self.model_name = config.model or "distilgpt2"
                self.temperature = getattr(config, "temperature", 0.7)
                self.max_tokens = getattr(config, "max_tokens", 256)
                self.device = getattr(config, "device", "cpu")
                self.load_in_8bit = getattr(config, "load_in_8bit", False)
                self.top_p = getattr(config, "top_p", 0.9)
        else:
            self.model_name = "distilgpt2"
            self.temperature = 0.7
            self.max_tokens = 256
            self.device = "cpu"
            self.load_in_8bit = False
            self.top_p = 0.9

        # Initialize model and tokenizer
        self.pipeline = None
        self.tokenizer = None
        # Lazy initialization - initialize only when needed

    def _initialize_model(self):
        """Initialize the model and tokenizer."""
        logger.info(f"Loading local model: {self.model_name}")

        try:
            # Use a simple generation pipeline for easier initialization
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device=-1,  # CPU
                truncation=True,
                max_length=512,
            )

            # Store tokenizer reference for token counting
            self.tokenizer = self.pipeline.tokenizer

            logger.info(f"Model {self.model_name} loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the local model.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        # Initialize on first use
        if self.pipeline is None:
            self._initialize_model()

        try:
            # Use pipeline for generation
            result = self.pipeline(
                prompt,
                max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                do_sample=kwargs.get("temperature", self.temperature) > 0,
                num_return_sequences=1,
            )

            # Extract generated text
            generated_text = result[0]["generated_text"]

            # Remove the input prompt from the output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt) :].strip()

            return generated_text

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error: {str(e)}"

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts.

        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters

        Returns:
            List of generated texts
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results

    def embed(self, text: str) -> List[float]:
        """Get embeddings for text (not implemented for generation models).

        Args:
            text: Input text

        Returns:
            Empty list (embeddings not supported)
        """
        logger.warning(f"Embeddings not supported for {self.model_name}")
        return []

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts (not implemented).

        Args:
            texts: List of input texts

        Returns:
            List of empty lists
        """
        return [[] for _ in texts]

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        # Initialize if needed
        if self.tokenizer is None:
            if self.pipeline is None:
                self._initialize_model()

        return len(self.tokenizer.encode(text))

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.

        Returns:
            Model information dictionary
        """
        return {
            "provider": "local",
            "model": self.model_name,
            "device": self.device,
            "load_in_8bit": self.load_in_8bit,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
