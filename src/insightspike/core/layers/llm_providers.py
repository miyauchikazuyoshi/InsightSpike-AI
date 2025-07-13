"""
Real LLM Providers for InsightSpike
===================================

Provides actual LLM functionality using OpenAI, Anthropic, or Ollama.
This module integrates with InsightSpike's Layer 4 architecture.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import handling with graceful fallbacks
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.info("OpenAI not installed. Install with: poetry install -E llm")

try:
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.info("Anthropic not installed. Install with: poetry install -E llm")

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.info("Ollama not installed. Install with: poetry install -E llm")


class RealOpenAIProvider:
    """
    OpenAI API-based LLM provider using official SDK.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI SDK not available. Install with: poetry install -E llm"
            )

        self.model_name = model_name
        self.temperature = temperature
        self._initialized = False
        self.client = None

    def initialize(self) -> bool:
        """Initialize OpenAI client"""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
            return False

        try:
            self.client = OpenAI(api_key=api_key)
            # Test connection
            self.client.models.list()
            self._initialized = True
            logger.info(f"OpenAI provider initialized with model: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            return False

    def generate_response(self, context: str, question: str) -> str:
        """Generate response using OpenAI"""
        if not self._initialized:
            if not self.initialize():
                return "Error: OpenAI not available"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on the provided context. Be concise and accurate.",
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}",
                    },
                ],
                temperature=self.temperature,
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return f"Error: {str(e)}"

    def generate_response_detailed(
        self, context: Dict[str, Any], question: str
    ) -> Dict[str, Any]:
        """Generate detailed response with metadata"""
        start_time = time.time()

        # Extract text context
        if isinstance(context, dict):
            docs = context.get("retrieved_documents", [])
            context_text = self._format_context(docs)
        else:
            context_text = str(context)

        response_text = self.generate_response(context_text, question)

        # Analyze response quality
        quality = self._estimate_quality(response_text, question)

        return {
            "response": response_text,
            "success": True,
            "confidence": quality["confidence"],
            "reasoning_quality": quality["reasoning_quality"],
            "insight_detected": quality["insight_detected"],
            "model_used": self.model_name,
            "processing_time": time.time() - start_time,
        }

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format documents into context string"""
        if not documents:
            return "No specific context available."

        context_parts = []
        for i, doc in enumerate(documents[:5]):
            text = doc.get("text", "")
            similarity = doc.get("similarity", 0.0)
            context_parts.append(f"[{i+1}] (relevance: {similarity:.2f}) {text}")

        return "\n\n".join(context_parts)

    def _estimate_quality(self, response: str, question: str) -> Dict[str, float]:
        """Estimate response quality metrics"""
        response_lower = response.lower()
        question_lower = question.lower()

        # Error detection
        if response.startswith("Error:"):
            return {
                "confidence": 0.0,
                "reasoning_quality": 0.0,
                "insight_detected": False,
            }

        # Length-based quality
        word_count = len(response.split())
        length_score = min(1.0, word_count / 100)

        # Keyword relevance
        question_words = set(question_lower.split())
        response_words = set(response_lower.split())
        relevance = len(question_words & response_words) / max(1, len(question_words))

        # Insight indicators
        insight_keywords = [
            "because",
            "therefore",
            "however",
            "although",
            "relationship",
            "connection",
            "implies",
            "suggests",
            "indicates",
            "reveals",
        ]
        insight_score = sum(1 for kw in insight_keywords if kw in response_lower) / len(
            insight_keywords
        )

        confidence = 0.6 + 0.3 * relevance + 0.1 * length_score
        reasoning_quality = 0.5 + 0.25 * length_score + 0.25 * insight_score

        return {
            "confidence": min(0.9, confidence),
            "reasoning_quality": min(0.85, reasoning_quality),
            "insight_detected": insight_score > 0.3,
        }


class RealAnthropicProvider:
    """
    Anthropic Claude API-based LLM provider using official SDK.
    """

    def __init__(
        self, model_name: str = "claude-3-sonnet-20240229", temperature: float = 0.7
    ):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic SDK not available. Install with: poetry install -E llm"
            )

        self.model_name = model_name
        self.temperature = temperature
        self._initialized = False
        self.client = None

    def initialize(self) -> bool:
        """Initialize Anthropic client"""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not found in environment variables")
            return False

        try:
            self.client = Anthropic(api_key=api_key)
            self._initialized = True
            logger.info(f"Anthropic provider initialized with model: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic: {e}")
            return False

    def generate_response(self, context: str, question: str) -> str:
        """Generate response using Anthropic"""
        if not self._initialized:
            if not self.initialize():
                return "Error: Anthropic not available"

        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": f"Based on the following context, please answer the question.\n\nContext:\n{context}\n\nQuestion: {question}",
                    }
                ],
                temperature=self.temperature,
                max_tokens=500,
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            return f"Error: {str(e)}"

    def generate_response_detailed(
        self, context: Dict[str, Any], question: str
    ) -> Dict[str, Any]:
        """Generate detailed response with metadata"""
        start_time = time.time()

        # Extract text context
        if isinstance(context, dict):
            docs = context.get("retrieved_documents", [])
            context_text = self._format_context(docs)
        else:
            context_text = str(context)

        response_text = self.generate_response(context_text, question)

        # Analyze response quality
        quality = self._estimate_quality(response_text, question)

        return {
            "response": response_text,
            "success": True,
            "confidence": quality["confidence"],
            "reasoning_quality": quality["reasoning_quality"],
            "insight_detected": quality["insight_detected"],
            "model_used": self.model_name,
            "processing_time": time.time() - start_time,
        }

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format documents into context string"""
        if not documents:
            return "No specific context available."

        context_parts = []
        for i, doc in enumerate(documents[:5]):
            text = doc.get("text", "")
            similarity = doc.get("similarity", 0.0)
            context_parts.append(f"[{i+1}] (relevance: {similarity:.2f}) {text}")

        return "\n\n".join(context_parts)

    def _estimate_quality(self, response: str, question: str) -> Dict[str, float]:
        """Estimate response quality metrics"""
        response_lower = response.lower()
        question_lower = question.lower()

        # Error detection
        if response.startswith("Error:"):
            return {
                "confidence": 0.0,
                "reasoning_quality": 0.0,
                "insight_detected": False,
            }

        # Length-based quality
        word_count = len(response.split())
        length_score = min(1.0, word_count / 100)

        # Keyword relevance
        question_words = set(question_lower.split())
        response_words = set(response_lower.split())
        relevance = len(question_words & response_words) / max(1, len(question_words))

        # Insight indicators
        insight_keywords = [
            "because",
            "therefore",
            "however",
            "although",
            "relationship",
            "connection",
            "implies",
            "suggests",
            "indicates",
            "reveals",
        ]
        insight_score = sum(1 for kw in insight_keywords if kw in response_lower) / len(
            insight_keywords
        )

        confidence = 0.6 + 0.3 * relevance + 0.1 * length_score
        reasoning_quality = 0.5 + 0.25 * length_score + 0.25 * insight_score

        return {
            "confidence": min(0.9, confidence),
            "reasoning_quality": min(0.85, reasoning_quality),
            "insight_detected": insight_score > 0.3,
        }


class RealOllamaProvider:
    """
    Ollama-based LLM provider for local inference.
    """

    def __init__(self, model_name: str = "llama2", temperature: float = 0.7):
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "Ollama SDK not available. Install with: poetry install -E llm"
            )

        self.model_name = model_name
        self.temperature = temperature
        self._initialized = False

    def initialize(self) -> bool:
        """Check if Ollama is available"""
        try:
            # List available models
            models = ollama.list()
            model_names = [m["name"] for m in models["models"]]

            if not model_names:
                logger.warning("No Ollama models installed. Run: ollama pull llama2")
                return False

            if self.model_name not in model_names:
                logger.warning(
                    f"Model {self.model_name} not found. Available: {model_names}"
                )
                if model_names:
                    self.model_name = model_names[0].split(":")[0]  # Use base name
                    logger.info(f"Using available model: {self.model_name}")

            self._initialized = True
            logger.info(f"Ollama provider initialized with model: {self.model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
            return False

    def generate_response(self, context: str, question: str) -> str:
        """Generate response using Ollama"""
        if not self._initialized:
            if not self.initialize():
                return "Error: Ollama not available"

        prompt = f"Based on the following context, please answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": self.temperature, "num_predict": 500},
            )
            return response["response"]
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return f"Error: {str(e)}"

    def generate_response_detailed(
        self, context: Dict[str, Any], question: str
    ) -> Dict[str, Any]:
        """Generate detailed response with metadata"""
        start_time = time.time()

        # Extract text context
        if isinstance(context, dict):
            docs = context.get("retrieved_documents", [])
            context_text = self._format_context(docs)
        else:
            context_text = str(context)

        response_text = self.generate_response(context_text, question)

        # Analyze response quality
        quality = self._estimate_quality(response_text, question)

        return {
            "response": response_text,
            "success": True,
            "confidence": quality["confidence"],
            "reasoning_quality": quality["reasoning_quality"],
            "insight_detected": quality["insight_detected"],
            "model_used": self.model_name,
            "processing_time": time.time() - start_time,
        }

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format documents into context string"""
        if not documents:
            return "No specific context available."

        context_parts = []
        for i, doc in enumerate(documents[:5]):
            text = doc.get("text", "")
            similarity = doc.get("similarity", 0.0)
            context_parts.append(f"[{i+1}] (relevance: {similarity:.2f}) {text}")

        return "\n\n".join(context_parts)

    def _estimate_quality(self, response: str, question: str) -> Dict[str, float]:
        """Estimate response quality metrics"""
        response_lower = response.lower()
        question_lower = question.lower()

        # Error detection
        if response.startswith("Error:"):
            return {
                "confidence": 0.0,
                "reasoning_quality": 0.0,
                "insight_detected": False,
            }

        # Length-based quality
        word_count = len(response.split())
        length_score = min(1.0, word_count / 100)

        # Keyword relevance
        question_words = set(question_lower.split())
        response_words = set(response_lower.split())
        relevance = len(question_words & response_words) / max(1, len(question_words))

        # Insight indicators
        insight_keywords = [
            "because",
            "therefore",
            "however",
            "although",
            "relationship",
            "connection",
            "implies",
            "suggests",
            "indicates",
            "reveals",
        ]
        insight_score = sum(1 for kw in insight_keywords if kw in response_lower) / len(
            insight_keywords
        )

        confidence = 0.6 + 0.3 * relevance + 0.1 * length_score
        reasoning_quality = 0.5 + 0.25 * length_score + 0.25 * insight_score

        return {
            "confidence": min(0.9, confidence),
            "reasoning_quality": min(0.85, reasoning_quality),
            "insight_detected": insight_score > 0.3,
        }


def get_real_llm_provider(provider_type: str = "auto") -> Any:
    """
    Get appropriate LLM provider with real implementations.

    Args:
        provider_type: "openai", "anthropic", "ollama", or "auto"

    Returns:
        LLM provider instance
    """
    # Check what's available
    available = []
    if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
        available.append("openai")
    if ANTHROPIC_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY"):
        available.append("anthropic")
    if OLLAMA_AVAILABLE:
        available.append("ollama")

    if not available:
        raise RuntimeError(
            "No LLM providers available. Please:\n"
            "1. Install LLM support: poetry install -E llm\n"
            "2. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable\n"
            "3. Or install and run Ollama: https://ollama.ai"
        )

    if provider_type == "openai":
        if "openai" not in available:
            raise RuntimeError("OpenAI requested but not available")
        return RealOpenAIProvider()

    elif provider_type == "anthropic":
        if "anthropic" not in available:
            raise RuntimeError("Anthropic requested but not available")
        return RealAnthropicProvider()

    elif provider_type == "ollama":
        if "ollama" not in available:
            raise RuntimeError("Ollama requested but not available")
        return RealOllamaProvider()

    elif provider_type == "auto":
        # Priority order: OpenAI > Anthropic > Ollama
        if "openai" in available:
            logger.info("Auto-selected OpenAI provider")
            return RealOpenAIProvider()
        elif "anthropic" in available:
            logger.info("Auto-selected Anthropic provider")
            return RealAnthropicProvider()
        elif "ollama" in available:
            logger.info("Auto-selected Ollama provider")
            return RealOllamaProvider()

    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


# Integration with InsightSpike's L4LLMProvider interface
class RealLLMProvider:
    """
    Adapter to make real LLMs compatible with InsightSpike's Layer 4 interface.
    """

    def __init__(self, provider_type: str = "auto"):
        self.provider = get_real_llm_provider(provider_type)
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the provider"""
        if hasattr(self.provider, "initialize"):
            self._initialized = self.provider.initialize()
        else:
            self._initialized = True
        return self._initialized

    def generate_response(self, context: Any, question: str) -> str:
        """Simple string response (L4LLMInterface)"""
        return self.provider.generate_response(context, question)

    def generate_response_detailed(
        self, context: Dict[str, Any], question: str
    ) -> Dict[str, Any]:
        """Detailed response with metadata"""
        return self.provider.generate_response_detailed(context, question)

    def cleanup(self):
        """Cleanup resources"""
        pass

    def format_context(self, episodes: List[Dict[str, Any]]) -> str:
        """Format episodes into context"""
        if hasattr(self.provider, "_format_context"):
            return self.provider._format_context(episodes)
        return str(episodes)

    def process(self, input_data) -> Dict[str, Any]:
        """Process input (legacy interface)"""
        if hasattr(input_data, "prompt"):
            prompt = input_data.prompt
            context = input_data.context if hasattr(input_data, "context") else ""
        else:
            prompt = str(input_data)
            context = ""

        result = self.generate_response_detailed({"context": context}, prompt)
        return {
            "result": result["response"],
            "confidence": result.get("confidence", 0.5),
            "metadata": result,
        }
