"""
Layer 4 Prompt Builder - Semantic Response Generation
=====================================================

True Layer 4 that generates structured responses from graph analysis.
This is the core semantic processing layer that transforms graph reasoning
into human-readable insights.
"""

import logging
from typing import Any, Dict, List, Optional

from ..interfaces.layer4_interface import L4Interface

logger = logging.getLogger(__name__)

__all__ = ["L4PromptBuilder"]


class L4PromptBuilder(L4Interface):
    """Layer 4: Semantic Response Generation

    This is the true Layer 4 that performs semantic processing:
    - Transforms graph analysis into structured responses
    - Synthesizes insights from retrieved documents
    - Generates human-readable explanations
    - Can work standalone (direct generation) or with LLM (polish mode)
    """

    def __init__(self, config=None):
        """Initialize with configuration object.

        Args:
            config: Configuration object (legacy format). If None, a default config will be used.
        """
        if config is None:
            # Create a minimal default config
            logger.warning("No config provided to L4PromptBuilder, using defaults")
            from types import SimpleNamespace

            config = SimpleNamespace(
                llm=SimpleNamespace(use_direct_generation=True, max_prompt_length=8000)
            )
        self.config = config

    def build_prompt(self, context: Dict[str, Any], question: str) -> str:
        """Build a comprehensive prompt with context and question."""
        try:
            # Extract context components
            documents = context.get("retrieved_documents", [])
            graph_info = context.get("graph_analysis", {})
            previous_state = context.get("previous_state", {})
            reasoning_quality = context.get("reasoning_quality", 0.0)

            # Build prompt sections
            sections = []

            # System instruction
            sections.append(self._build_system_instruction())

            # Context documents
            if documents:
                sections.append(self._build_document_context(documents))

            # Reasoning state
            if graph_info:
                sections.append(self._build_reasoning_context(graph_info))

            # Previous insights
            if previous_state:
                sections.append(self._build_previous_context(previous_state))

            # Question and instructions
            sections.append(self._build_question_section(question, reasoning_quality))

            return "\n\n".join(sections)

        except Exception as e:
            logger.error(f"Prompt building failed: {e}")
            return self._fallback_prompt(question)

    def _build_system_instruction(self) -> str:
        """Build system instruction section."""
        return """You are an advanced AI assistant specialized in analytical reasoning and insight generation. 

Your role is to:
1. Analyze provided documents and context carefully
2. Identify key patterns, connections, and insights
3. Provide well-reasoned answers based on evidence
4. Acknowledge uncertainty when information is insufficient
5. Highlight novel insights or "spikes of understanding" when they emerge

Always base your responses on the provided context and clearly distinguish between what the evidence supports versus speculative reasoning."""

    def _build_document_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build document context section."""
        context_lines = ["## Retrieved Context Documents"]
        context_lines.append("The following documents are relevant to your query:\n")

        for i, doc in enumerate(documents[:5], 1):  # Limit to top 5 documents
            text = doc.get("text", "").strip()
            c_value = doc.get("c_value", 0.0)
            similarity = doc.get("similarity", 0.0)

            if text:
                confidence_indicator = self._get_confidence_indicator(
                    c_value, similarity
                )
                context_lines.append(f"### Document {i} {confidence_indicator}")
                context_lines.append(
                    f"**Relevance:** {similarity:.3f} | **Confidence:** {c_value:.3f}"
                )
                context_lines.append(f"{text}\n")

        return "\n".join(context_lines)

    def _build_reasoning_context(self, graph_info: Dict[str, Any]) -> str:
        """Build reasoning state context."""
        lines = ["## Current Reasoning State"]

        metrics = graph_info.get("metrics", {})
        if metrics:
            lines.append("**Graph Analysis Metrics:**")
            lines.append(
                f"- ΔGED (Graph Edit Distance Change): {metrics.get('delta_ged', 0):.3f}"
            )
            lines.append(
                f"- ΔIG (Information Gain Change): {metrics.get('delta_ig', 0):.3f}"
            )

        conflicts = graph_info.get("conflicts", {})
        if conflicts:
            total_conflict = conflicts.get("total", 0)
            lines.append(f"- Conflict Level: {total_conflict:.3f}")

        spike_detected = graph_info.get("spike_detected", False)
        if spike_detected:
            lines.append(
                "\n🧠 **INSIGHT SPIKE DETECTED** - This query may represent a significant improvement in understanding!"
            )

        reasoning_quality = graph_info.get("reasoning_quality", 0)
        if reasoning_quality > 0:
            lines.append(f"- Reasoning Quality Score: {reasoning_quality:.3f}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def _build_previous_context(self, previous_state: Dict[str, Any]) -> str:
        """Build previous reasoning context."""
        lines = ["## Previous Context"]

        if "last_response" in previous_state:
            lines.append("**Previous Response:**")
            lines.append(f"{previous_state['last_response'][:200]}...")

        if "reasoning_chain" in previous_state:
            lines.append("**Reasoning Chain:**")
            chain = previous_state["reasoning_chain"]
            if isinstance(chain, list):
                for step in chain[-3:]:  # Last 3 steps
                    lines.append(f"- {step}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def _build_question_section(
        self, question: str, reasoning_quality: float = 0.0
    ) -> str:
        """Build question and instruction section."""
        lines = ["## User Question"]
        lines.append(f'"{question}"')
        lines.append("")
        lines.append("## Instructions")

        if reasoning_quality > 0.7:
            lines.append(
                "High reasoning quality detected. Provide a comprehensive, well-structured answer."
            )
        elif reasoning_quality > 0.4:
            lines.append(
                "Moderate reasoning quality. Provide a balanced answer with appropriate caveats."
            )
        else:
            lines.append(
                "Limited context available. Provide what insights you can while acknowledging limitations."
            )

        lines.append("")
        lines.append("Please:")
        lines.append("1. Synthesize information from the provided context")
        lines.append("2. Highlight key insights and connections")
        lines.append("3. Indicate confidence levels in your reasoning")
        lines.append("4. Note any novel patterns or 'insight spikes' you detect")
        lines.append("5. Provide a clear, actionable answer")

        return "\n".join(lines)

    def _get_confidence_indicator(self, c_value: float, similarity: float) -> str:
        """Get visual indicator for document confidence."""
        combined_score = (c_value + similarity) / 2

        if combined_score > 0.8:
            return "🟢 (High Confidence)"
        elif combined_score > 0.6:
            return "🟡 (Medium Confidence)"
        elif combined_score > 0.4:
            return "🟠 (Low Confidence)"
        else:
            return "🔴 (Very Low Confidence)"

    def _fallback_prompt(self, question: str) -> str:
        """Fallback prompt when building fails."""
        return f"""Please answer the following question based on your knowledge:

Question: {question}

Please provide a clear, well-reasoned response."""

    def build_simple_prompt(self, documents: List[str], question: str) -> str:
        """Build a simple prompt without advanced context."""
        if not documents:
            return f"Question: {question}\n\nPlease answer based on your knowledge."

        context = "\n\n".join(
            f"Document {i+1}: {doc}" for i, doc in enumerate(documents[:3])
        )

        return f"""Context Documents:
{context}

Question: {question}

Please answer the question based on the provided context documents."""

    def build_direct_response(self, context: Dict[str, Any], question: str) -> str:
        """Build a direct response without LLM processing.

        This method generates a complete, structured response based on the
        context and reasoning analysis, suitable for high-confidence scenarios.
        """
        try:
            # Extract context components
            documents = context.get("retrieved_documents", [])
            graph_info = context.get("graph_analysis", {})
            reasoning_quality = context.get("reasoning_quality", 0.0)

            # Start building the response
            response_parts = []

            # Add insight spike notification if detected
            if graph_info.get("spike_detected", False):
                response_parts.append("🧠 **INSIGHT SPIKE DETECTED**")
                response_parts.append("")

            # Main answer section
            response_parts.append("## Answer")

            # Synthesize information from documents
            if documents:
                key_points = self._extract_key_points(documents)
                response_parts.append(self._synthesize_answer(key_points, question))
            else:
                response_parts.append(
                    "Based on the analysis of the knowledge graph structure:"
                )
                response_parts.append("")

            # Add reasoning metrics if significant
            metrics = graph_info.get("metrics", {})
            if metrics:
                delta_ged = metrics.get("delta_ged", 0)
                delta_ig = metrics.get("delta_ig", 0)

                if abs(delta_ged) > 0.1 or abs(delta_ig) > 0.1:
                    response_parts.append("")
                    response_parts.append("## Reasoning Metrics")
                    response_parts.append(
                        f"- Structure simplification (ΔGED): {delta_ged:.3f}"
                    )
                    response_parts.append(f"- Information gain (ΔIG): {delta_ig:.3f}")

            # Add confidence level
            response_parts.append("")
            response_parts.append(
                f"**Confidence Level**: {self._get_confidence_indicator(reasoning_quality, reasoning_quality)}"
            )

            # Add synthesis note if cross-domain connections detected
            if reasoning_quality > 0.7:
                response_parts.append("")
                response_parts.append(
                    "*This response synthesizes information across multiple knowledge domains.*"
                )

            return "\n".join(response_parts)

        except Exception as e:
            logger.error(f"Direct response building failed: {e}")
            return self._fallback_response(question)

    def _extract_key_points(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Extract key points from documents."""
        key_points = []
        for doc in documents[:5]:  # Top 5 documents
            text = doc.get("text", "").strip()
            if text:
                # Extract first sentence or key phrase
                sentences = text.split(". ")
                if sentences:
                    key_points.append(sentences[0] + ".")
        return key_points

    def _synthesize_answer(self, key_points: List[str], question: str) -> str:
        """Synthesize key points into a coherent answer."""
        if not key_points:
            return "No relevant information was found in the knowledge base."

        # Simple synthesis - join key points with transitions
        synthesis = []
        for i, point in enumerate(key_points):
            if i == 0:
                synthesis.append(point)
            elif i < len(key_points) - 1:
                synthesis.append(f"Additionally, {point.lower()}")
            else:
                synthesis.append(f"Furthermore, {point.lower()}")

        return " ".join(synthesis)

    def _fallback_response(self, question: str) -> str:
        """Fallback response for direct generation."""
        return f"Unable to generate a direct response for: {question}\n\nPlease try using standard LLM generation mode."

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through Layer 4.

        Implements the L4Interface process method.
        """
        context = input_data.get("context", {})
        question = input_data.get("question", "")
        mode = input_data.get("mode", None)

        # Determine generation mode if not specified
        if mode is None:
            mode = self.get_generation_mode(context)

        # Generate output based on mode
        if mode == "direct":
            output = self.build_direct_response(context, question)
        else:
            output = self.build_prompt(context, question)

        # Calculate confidence
        reasoning_quality = context.get("reasoning_quality", 0.0)

        return {
            "output": output,
            "mode": mode,
            "confidence": reasoning_quality,
            "metadata": {
                "spike_detected": context.get("graph_analysis", {}).get(
                    "spike_detected", False
                ),
                "metrics": context.get("graph_analysis", {}).get("metrics", {}),
            },
        }

    def get_generation_mode(self, context: Dict[str, Any]) -> str:
        """Determine the appropriate generation mode.

        Returns 'direct' for high-quality reasoning, 'prompt' otherwise.
        """
        reasoning_quality = context.get("reasoning_quality", 0.0)

        # Check if direct generation is enabled in config
        use_direct = getattr(self.config.llm, "use_direct_generation", False)
        threshold = getattr(self.config.llm, "direct_generation_threshold", 0.7)

        if use_direct and reasoning_quality > threshold:
            return "direct"
        return "prompt"
