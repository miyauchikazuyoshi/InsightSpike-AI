"""
Prompt Builder - Enhanced Prompt Construction for LLM Queries
==========================================================

Builds contextual prompts with retrieved documents and reasoning state.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["PromptBuilder"]


class PromptBuilder:
    """Builds enhanced prompts for LLM queries with context integration."""

    def __init__(self, config=None):
        from ..config import get_config

        self.config = config or get_config()

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
