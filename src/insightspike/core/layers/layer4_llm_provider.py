"""
Layer 4.1 LLM Polish - Optional Text Enhancement Layer
====================================================

Optional Layer 4.1 that polishes the structured responses from Layer 4 (PromptBuilder).
When direct generation mode is enabled, this layer can be bypassed entirely.

Architecture:
- Layer 4 (L4PromptBuilder): Core semantic generation
- Layer 4.1 (L4LLMProvider): Optional text polishing
"""

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ...config import get_config
from .layer4_prompt_builder import L4PromptBuilder
from ..interfaces import L4LLMInterface, LayerInput, LayerOutput

logger = logging.getLogger(__name__)

__all__ = ["L4LLMProvider", "OpenAIProvider", "LocalProvider", "get_llm_provider"]


class L4LLMProvider(L4LLMInterface):
    """Base class for LLM providers."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.prompt_builder = L4PromptBuilder(config)  # Using Layer 4 directly
        self._initialized = False

    def generate_response(self, context: str, question: str) -> str:
        """Generate response using the LLM."""
        if not self._initialized:
            self.initialize()

        try:
            # Parse context if it's a dict (for backward compatibility)
            if isinstance(context, dict):
                reasoning_quality = context.get("reasoning_quality", 0.0)
                context_str = str(context)
            else:
                reasoning_quality = 0.5
                context_str = context

            # Build enhanced prompt
            prompt = self.prompt_builder.build_prompt(
                {"context": context_str, "reasoning_quality": reasoning_quality},
                question,
            )

            # Check if we should use direct generation
            use_direct_generation = (
                hasattr(self.config, "llm")
                and hasattr(self.config.llm, "use_direct_generation")
                and self.config.llm.use_direct_generation
                and reasoning_quality
                > getattr(self.config.llm, "direct_generation_threshold", 0.7)
            )

            if use_direct_generation:
                # Use PromptBuilder to generate a complete response
                logger.info(
                    f"Using direct generation (reasoning_quality={reasoning_quality:.3f})"
                )

                # Generate direct response
                direct_response = self.prompt_builder.build_direct_response(
                    context, question
                )

                # Analyze the response for insights
                insight_analysis = self.analyze_insight_potential(
                    question, direct_response
                )

                return direct_response

            # Generate response
            response = self._generate_sync(prompt)
            return response

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating response: {str(e)}"

    def generate_response_detailed(
        self, context: Dict[str, Any], question: str, streaming: bool = False
    ) -> Dict[str, Any]:
        """Generate detailed response with metadata (for internal use)."""
        if not self._initialized:
            self.initialize()

        try:
            # Build enhanced prompt
            prompt = self.prompt_builder.build_prompt(context, question)

            # Check if we should use direct generation
            reasoning_quality = context.get("reasoning_quality", 0.0)
            use_direct_generation = (
                hasattr(self.config, "llm")
                and hasattr(self.config.llm, "use_direct_generation")
                and self.config.llm.use_direct_generation
                and reasoning_quality
                > getattr(self.config.llm, "direct_generation_threshold", 0.7)
                and not streaming  # Direct generation doesn't support streaming yet
            )

            if use_direct_generation:
                # Use PromptBuilder to generate a complete response
                logger.info(
                    f"Using direct generation (reasoning_quality={reasoning_quality:.3f})"
                )

                # Generate direct response
                direct_response = self.prompt_builder.build_direct_response(
                    context, question
                )

                # Analyze the response for insights
                insight_analysis = self.analyze_insight_potential(
                    question, direct_response
                )

                return {
                    "response": direct_response,
                    "prompt": prompt,
                    "streaming": False,
                    "success": True,
                    "confidence": reasoning_quality,
                    "reasoning_quality": reasoning_quality,
                    "insight_analysis": insight_analysis,
                    "synthesis_detected": insight_analysis["synthesis_detected"],
                    "cross_domain_score": insight_analysis["cross_domain_score"],
                    "direct_generation": True,
                }

            # Generate response
            if streaming:
                response_iter = self._generate_streaming(prompt)
                return {
                    "response": response_iter,
                    "prompt": prompt,
                    "streaming": True,
                    "success": True,
                }
            else:
                response = self._generate_sync(prompt)
                confidence = self._estimate_confidence(response, context)

                # Enhanced insight analysis
                insight_analysis = self.analyze_insight_potential(question, response)

                return {
                    "response": response,
                    "prompt": prompt,
                    "streaming": False,
                    "success": True,
                    "confidence": confidence,
                    "reasoning_quality": confidence * 0.8
                    + insight_analysis["insight_potential"] * 0.2,
                    "insight_analysis": insight_analysis,
                    "synthesis_detected": insight_analysis["synthesis_detected"],
                    "cross_domain_score": insight_analysis["cross_domain_score"],
                }

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your question.",
                "error": str(e),
                "success": False,
            }

    def initialize(self) -> bool:
        """Initialize the LLM provider."""
        raise NotImplementedError

    def _generate_sync(self, prompt: str) -> str:
        """Generate response synchronously."""
        raise NotImplementedError

    def _generate_streaming(self, prompt: str) -> Iterator[str]:
        """Generate response with streaming."""
        # Fallback to sync generation
        response = self._generate_sync(prompt)
        for char in response:
            yield char

    def _estimate_confidence(self, response: str, context: Dict[str, Any]) -> float:
        """Enhanced confidence estimation with insight analysis."""

        # Length-based confidence (very short or very long responses are less confident)
        length_score = min(1.0, len(response) / 200) * min(
            1.0, 500 / max(len(response), 1)
        )

        # Context relevance (if we have relevant documents)
        relevance_score = 1.0
        if "retrieved_documents" in context and context["retrieved_documents"]:
            num_docs = len(context["retrieved_documents"])
            relevance_score = min(1.0, num_docs / 3)  # More docs = higher confidence

        # Insight indicators boost confidence
        insight_indicators = [
            "by connecting",
            "synthesis",
            "insight emerges",
            "key insight",
            "bridging",
            "integrating",
            "cross-domain",
            "framework",
        ]
        insight_boost = sum(
            0.05 for indicator in insight_indicators if indicator in response.lower()
        )
        insight_boost = min(0.3, insight_boost)  # Cap the boost

        # Uncertainty indicators reduce confidence
        uncertainty_keywords = [
            "maybe",
            "perhaps",
            "might",
            "could",
            "uncertain",
            "not sure",
        ]
        uncertainty_count = sum(
            1 for word in uncertainty_keywords if word in response.lower()
        )
        uncertainty_penalty = min(0.5, uncertainty_count * 0.1)

        # Synthesis quality indicators
        synthesis_indicators = [
            "connecting multiple",
            "systematic analysis",
            "demonstrates that",
            "emerges from recognizing",
            "reveals that",
            "by examining",
        ]
        synthesis_boost = sum(
            0.08 for indicator in synthesis_indicators if indicator in response.lower()
        )
        synthesis_boost = min(0.2, synthesis_boost)

        confidence = (
            (length_score + relevance_score) / 2
            + insight_boost
            + synthesis_boost
            - uncertainty_penalty
        )
        return max(0.0, min(1.0, confidence))

    def analyze_insight_potential(self, question: str, response: str) -> Dict[str, Any]:
        """Analyze whether the response demonstrates insight or synthesis."""

        question_lower = question.lower()
        response_lower = response.lower()

        # Cross-domain indicators
        domain_keywords = {
            "probability": ["probability", "conditional", "bayes"],
            "mathematics": ["infinite", "convergence", "series", "limit"],
            "philosophy": ["identity", "criteria", "existence", "continuity"],
            "physics": ["quantum", "measurement", "uncertainty", "reality"],
            "information": ["information", "entropy", "asymmetric"],
            "systems": ["emergence", "complexity", "feedback", "non-linear"],
        }

        # Count domains mentioned
        domains_in_question = sum(
            1
            for domain, keywords in domain_keywords.items()
            if any(kw in question_lower for kw in keywords)
        )
        domains_in_response = sum(
            1
            for domain, keywords in domain_keywords.items()
            if any(kw in response_lower for kw in keywords)
        )

        # Synthesis indicators
        synthesis_patterns = [
            "by connecting",
            "by synthesizing",
            "by integrating",
            "synthesis emerges",
            "insight emerges",
            "key insight",
            "bridging",
            "connecting multiple",
            "cross-domain",
        ]
        synthesis_detected = any(
            pattern in response_lower for pattern in synthesis_patterns
        )

        # Calculate insight metrics
        cross_domain_score = min(1.0, domains_in_response / max(domains_in_question, 1))
        synthesis_score = 1.0 if synthesis_detected else 0.0

        return {
            "cross_domain_score": cross_domain_score,
            "synthesis_score": synthesis_score,
            "domains_mentioned": domains_in_response,
            "synthesis_detected": synthesis_detected,
            "insight_potential": (cross_domain_score + synthesis_score) / 2,
        }


class OpenAIProvider(L4LLMProvider):
    """OpenAI GPT provider."""

    def initialize(self) -> bool:
        """Initialize OpenAI client."""
        try:
            import openai

            api_key = self.config.llm.openai_api_key
            if not api_key:
                logger.error("OpenAI API key not found in config")
                return False

            self.client = openai.OpenAI(api_key=api_key)
            self.model = self.config.llm.openai_model
            self._initialized = True

            logger.info(f"Initialized OpenAI provider with model: {self.model}")
            return True

        except ImportError:
            logger.error("OpenAI package not installed. Run: pip install openai")
            return False
        except Exception as e:
            logger.error(f"OpenAI initialization failed: {e}")
            return False

    def _generate_sync(self, prompt: str) -> str:
        """Generate response using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant specialized in answering questions based on provided context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

    def _generate_streaming(self, prompt: str) -> Iterator[str]:
        """Generate streaming response using OpenAI API."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant specialized in answering questions based on provided context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            yield f"Error: {e}"

    def format_context(self, episodes: List[Dict[str, Any]]) -> str:
        """Format episodes into context string."""
        if not episodes:
            return ""

        context_parts = []
        for i, episode in enumerate(
            episodes[:10]
        ):  # Limit to 10 most relevant episodes
            text = episode.get("text", str(episode))
            c_value = episode.get("c", 0.5)
            context_parts.append(f"Context {i+1} (relevance: {c_value:.2f}):\n{text}")

        return "\n\n".join(context_parts)

    def process(self, input_data: LayerInput) -> LayerOutput:
        """Process input through LLM layer."""
        if not isinstance(input_data, LayerInput):
            # Convert dict to LayerInput for backward compatibility
            if isinstance(input_data, dict):
                input_data = LayerInput(
                    data=input_data.get("question", str(input_data)),
                    context=input_data.get("context", {}),
                    metadata=input_data.get("metadata", {}),
                )
            else:
                input_data = LayerInput(data=str(input_data))

        context = input_data.context or {}
        question = input_data.data

        # Convert context to string for generate_response
        if isinstance(context, dict):
            context_str = self.format_context(context.get("episodes", []))
        else:
            context_str = str(context)

        # Generate response (returns string)
        response = self.generate_response(context_str, question)

        # Create LayerOutput with enhanced metadata
        return LayerOutput(
            data=response,
            confidence=self.calculate_confidence(question, response),
            layer_name="L4_LLM",
            processing_time=0.0,  # Will be set by decorator if used
            metadata={
                "model": self.model,
                "provider": "openai",
                "question": question,
                "context_length": len(context_str),
                "response_length": len(response),
                "insight_analysis": self.analyze_insight_potential(question, response),
            },
        )

    def cleanup(self):
        """Cleanup OpenAI provider resources."""
        try:
            # OpenAI client doesn't need explicit cleanup
            self._initialized = False
            self.client = None
            logger.info("OpenAI provider cleaned up")
        except Exception as e:
            logger.error(f"Error during OpenAI cleanup: {e}")


class LocalProvider(L4LLMProvider):
    """Local model provider using transformers."""

    def initialize(self) -> bool:
        """Initialize local model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            model_name = self.config.llm.model_name

            logger.info(f"Loading local model: {model_name}")

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="auto"
                if hasattr(self.config.llm, "use_gpu") and self.config.llm.use_gpu
                else "cpu",
            )

            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                do_sample=True if self.config.llm.temperature > 0 else False,
            )

            self._initialized = True
            logger.info(f"Initialized local model: {model_name}")
            return True

        except ImportError:
            logger.error(
                "Transformers package not installed. Run: pip install transformers torch"
            )
            return False
        except Exception as e:
            logger.error(f"Local model initialization failed: {e}")
            return False

    def _generate_sync(self, prompt: str) -> str:
        """Generate response using local model."""
        try:
            # Add instruction formatting for better results
            formatted_prompt = self._format_prompt(prompt)

            outputs = self.pipeline(
                formatted_prompt,
                max_new_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                do_sample=True if self.config.llm.temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            generated_text = outputs[0]["generated_text"]

            # Extract only the new part (after the prompt)
            response = generated_text[len(formatted_prompt) :].strip()

            # Clean up common artifacts
            response = self._clean_response(response)

            return response

        except Exception as e:
            logger.error(f"Local generation failed: {e}")
            raise

    def _format_prompt(self, prompt: str) -> str:
        """Format prompt for better local model performance."""
        return f"""<|system|>
You are a helpful AI assistant. Answer the question based on the provided context.

<|user|>
{prompt}

<|assistant|>
"""

    def format_context(self, episodes: List[Dict[str, Any]]) -> str:
        """Format episodes into context string."""
        if not episodes:
            return ""

        context_parts = []
        for i, episode in enumerate(
            episodes[:10]
        ):  # Limit to 10 most relevant episodes
            text = episode.get("text", str(episode))
            c_value = episode.get("c", 0.5)
            context_parts.append(f"Context {i+1} (relevance: {c_value:.2f}):\n{text}")

        return "\n\n".join(context_parts)

    def process(self, input_data: LayerInput) -> LayerOutput:
        """Process input through LLM layer."""
        if not isinstance(input_data, LayerInput):
            # Convert dict to LayerInput for backward compatibility
            if isinstance(input_data, dict):
                input_data = LayerInput(
                    data=input_data.get("question", str(input_data)),
                    context=input_data.get("context", {}),
                    metadata=input_data.get("metadata", {}),
                )
            else:
                input_data = LayerInput(data=str(input_data))

        context = input_data.context or {}
        question = input_data.data

        # Convert context to string for generate_response
        if isinstance(context, dict):
            context_str = self.format_context(context.get("episodes", []))
        else:
            context_str = str(context)

        # Generate response (returns string)
        response = self.generate_response(context_str, question)

        # For detailed metadata, use the internal method
        if hasattr(self, "generate_response_detailed"):
            detailed_result = self.generate_response_detailed(
                input_data.context or {}, question
            )
            return LayerOutput(
                result=response,
                confidence=detailed_result.get("confidence", 0.8),
                metadata=detailed_result,
            )
        else:
            return LayerOutput(
                result=response,
                confidence=0.8,
                metadata={"response": response, "success": True},
            )

    def cleanup(self):
        """Cleanup resources."""
        try:
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "tokenizer"):
                del self.tokenizer
            if hasattr(self, "pipeline"):
                del self.pipeline

            # Clear GPU memory if using CUDA
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("LocalProvider cleanup completed")
        except Exception as e:
            logger.warning(f"Error during LocalProvider cleanup: {e}")

    def _clean_response(self, response: str) -> str:
        """Clean up response artifacts."""
        # Remove common stop sequences
        stop_sequences = ["<|", "</s>", "<s>", "[END]", "[STOP]"]

        for stop in stop_sequences:
            if stop in response:
                response = response.split(stop)[0]

        # Remove excessive whitespace
        response = response.strip()

        # Remove incomplete sentences at the end
        sentences = response.split(". ")
        if len(sentences) > 1 and not sentences[-1].endswith("."):
            response = ". ".join(sentences[:-1]) + "."

        return response


def get_llm_provider(config=None, safe_mode=False) -> L4LLMProvider:
    """Get appropriate LLM provider based on configuration."""
    config = config or get_config()

    # Check for safe mode from config or parameter
    use_safe_mode = (
        safe_mode
        or getattr(config.llm, "safe_mode", False)
        or getattr(config, "environment", "local") == "testing"
    )

    if use_safe_mode:
        from .clean_llm_provider import CleanLLMProvider

        logger.info("Using clean LLM provider for safe operation (no data leaks)")
        return CleanLLMProvider(config)

    provider_type = config.llm.provider.lower()

    if provider_type == "openai":
        return OpenAIProvider(config)
    elif provider_type == "local":
        return LocalProvider(config)
    else:
        logger.warning(f"Unknown provider type: {provider_type}, falling back to local")
        return LocalProvider(config)


# Legacy compatibility function
def generate(prompt: str) -> str:
    """Legacy generate function for backward compatibility."""
    try:
        provider = get_llm_provider()
        result = provider.generate_response("", prompt)
        return result

    except Exception as e:
        logger.error(f"Legacy generate failed: {e}")
        return "Error generating response."
