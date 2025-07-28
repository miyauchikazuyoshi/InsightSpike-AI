"""
Response Generator
=================

Handles response formatting and output generation.
Separated from MainAgent to follow Single Responsibility Principle.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .structures import CycleResult
from ..interfaces import ILLMProvider

logger = logging.getLogger(__name__)


@dataclass
class ResponseTemplate:
    """Template for response formatting."""
    
    question: str
    answer: str
    reasoning: Optional[str] = None
    confidence: float = 0.0
    sources: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class ResponseGenerator:
    """
    Generates and formats responses for user queries.
    
    Responsibilities:
    - Format final responses
    - Apply response templates
    - Handle multi-modal outputs
    - Manage response quality
    """
    
    def __init__(
        self,
        llm_provider: ILLMProvider,
        response_style: str = "concise"
    ):
        """
        Initialize response generator.
        
        Args:
            llm_provider: LLM provider for generation
            response_style: Response formatting style
        """
        self.llm_provider = llm_provider
        self.response_style = response_style
        
        self._response_templates = {
            "concise": self._format_concise,
            "detailed": self._format_detailed,
            "academic": self._format_academic,
            "conversational": self._format_conversational
        }
    
    def generate_final_response(
        self,
        cycle_results: List[CycleResult],
        question: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Generate final response from reasoning cycles.
        
        Args:
            cycle_results: Results from reasoning cycles
            question: Original question
            context: Additional context
            
        Returns:
            Formatted response string
        """
        try:
            # Get best result
            best_result = self._select_best_result(cycle_results)
            
            # Create response template
            template = ResponseTemplate(
                question=question,
                answer=best_result.response,
                reasoning=best_result.reasoning_trace,
                confidence=best_result.reasoning_quality,
                sources=self._extract_sources(best_result),
                metadata={
                    "cycles": len(cycle_results),
                    "spike_detected": best_result.spike_detected,
                    "convergence": best_result.convergence_score
                }
            )
            
            # Format according to style
            formatter = self._response_templates.get(
                self.response_style,
                self._format_concise
            )
            
            return formatter(template)
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return self._error_response(str(e))
    
    def format_intermediate_response(
        self,
        cycle_result: CycleResult,
        cycle_num: int
    ) -> str:
        """
        Format intermediate cycle response for verbose mode.
        
        Args:
            cycle_result: Single cycle result
            cycle_num: Cycle number
            
        Returns:
            Formatted intermediate response
        """
        parts = [
            f"=== Cycle {cycle_num} ===",
            f"Response: {cycle_result.response}",
            f"Quality: {cycle_result.reasoning_quality:.3f}",
            f"Spike Detected: {cycle_result.spike_detected}"
        ]
        
        if cycle_result.reasoning_trace:
            parts.append(f"Reasoning: {cycle_result.reasoning_trace[:200]}...")
        
        return "\n".join(parts)
    
    def generate_summary(
        self,
        cycle_results: List[CycleResult],
        question: str
    ) -> Dict[str, Any]:
        """
        Generate execution summary.
        
        Args:
            cycle_results: All cycle results
            question: Original question
            
        Returns:
            Summary dictionary
        """
        spike_cycles = [i for i, r in enumerate(cycle_results) if r.spike_detected]
        quality_scores = [r.reasoning_quality for r in cycle_results]
        
        return {
            "question": question,
            "total_cycles": len(cycle_results),
            "spike_cycles": spike_cycles,
            "final_answer": cycle_results[-1].response if cycle_results else None,
            "average_quality": sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
            "max_quality": max(quality_scores) if quality_scores else 0.0,
            "convergence_achieved": cycle_results[-1].convergence_score > 0.9 if cycle_results else False
        }
    
    def _select_best_result(self, results: List[CycleResult]) -> CycleResult:
        """Select best result from cycles."""
        if not results:
            raise ValueError("No results to select from")
        
        # Prioritize spike-detected results with high quality
        spike_results = [r for r in results if r.spike_detected]
        if spike_results:
            return max(spike_results, key=lambda r: r.reasoning_quality)
        
        # Otherwise, return highest quality result
        return max(results, key=lambda r: r.reasoning_quality)
    
    def _extract_sources(self, result: CycleResult) -> List[str]:
        """Extract source references from result."""
        sources = []
        
        for memory_item in result.memory_used:
            if isinstance(memory_item, dict):
                text = memory_item.get('text', '')
                if text:
                    sources.append(text[:100] + "..." if len(text) > 100 else text)
        
        return sources[:5]  # Limit to top 5 sources
    
    def _format_concise(self, template: ResponseTemplate) -> str:
        """Format concise response."""
        return template.answer
    
    def _format_detailed(self, template: ResponseTemplate) -> str:
        """Format detailed response with reasoning."""
        parts = [
            f"Question: {template.question}",
            f"\nAnswer: {template.answer}"
        ]
        
        if template.reasoning:
            parts.append(f"\nReasoning: {template.reasoning}")
        
        if template.sources:
            parts.append("\nSources:")
            for i, source in enumerate(template.sources, 1):
                parts.append(f"  {i}. {source}")
        
        if template.confidence > 0:
            parts.append(f"\nConfidence: {template.confidence:.2%}")
        
        return "\n".join(parts)
    
    def _format_academic(self, template: ResponseTemplate) -> str:
        """Format academic-style response."""
        parts = [
            f"Query: {template.question}",
            f"\nResponse: {template.answer}"
        ]
        
        if template.reasoning:
            parts.append(f"\nMethodology: {template.reasoning}")
        
        if template.sources:
            parts.append("\nReferences:")
            for i, source in enumerate(template.sources, 1):
                parts.append(f"  [{i}] {source}")
        
        if template.metadata:
            spike = template.metadata.get('spike_detected', False)
            if spike:
                parts.append("\nNote: Novel insight detected during reasoning process.")
        
        return "\n".join(parts)
    
    def _format_conversational(self, template: ResponseTemplate) -> str:
        """Format conversational response."""
        confidence_phrases = {
            (0.9, 1.0): "I'm quite confident that",
            (0.7, 0.9): "I believe",
            (0.5, 0.7): "It seems that",
            (0.0, 0.5): "I'm not entirely sure, but"
        }
        
        # Find appropriate confidence phrase
        conf_phrase = "I think"
        for (low, high), phrase in confidence_phrases.items():
            if low <= template.confidence < high:
                conf_phrase = phrase
                break
        
        response = f"{conf_phrase} {template.answer}"
        
        if template.metadata and template.metadata.get('spike_detected'):
            response += "\n\nInterestingly, I discovered a new connection while thinking about this!"
        
        return response
    
    def _error_response(self, error_msg: str) -> str:
        """Generate error response."""
        return f"I encountered an error while processing your question: {error_msg}"