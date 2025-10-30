"""
Reasoning Engine
===============

Manages the reasoning cycle and spike detection logic.
Separated from MainAgent to follow Single Responsibility Principle.
"""

import logging
from typing import Dict, Any, List, Optional

from .structures import CycleResult
from ..interfaces import ILLMProvider, IMemorySearch
from ..implementations.layers import L3GraphReasoner, L4LLMInterface

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """
    Handles the core reasoning cycle logic.
    
    Responsibilities:
    - Execute reasoning cycles
    - Detect convergence
    - Manage spike detection
    """
    
    def __init__(
        self,
        llm_provider: ILLMProvider,
        memory_search: IMemorySearch,
        graph_reasoner: L3GraphReasoner,
        llm_interface: L4LLMInterface
    ):
        """
        Initialize reasoning engine with dependencies.
        
        Args:
            llm_provider: LLM provider for generation
            memory_search: Memory search component
            graph_reasoner: Graph reasoning layer
            llm_interface: LLM interface layer
        """
        self.llm_provider = llm_provider
        self.memory_search = memory_search
        self.graph_reasoner = graph_reasoner
        self.llm_interface = llm_interface
        
        self._convergence_threshold = 0.95
        self._min_quality_threshold = 0.8
    
    def execute_cycle(
        self,
        question: str,
        context: Dict[str, Any],
        verbose: bool = False
    ) -> CycleResult:
        """
        Execute a single reasoning cycle.
        
        Args:
            question: Question to process
            context: Current context including memory
            verbose: Whether to log details
            
        Returns:
            CycleResult for this cycle
        """
        try:
            # Phase 1: Memory retrieval with graph search
            memory_results = self._retrieve_relevant_memory(question, context)
            
            # Phase 2: Graph reasoning and spike detection
            graph_analysis = self._analyze_graph_state(memory_results, context)
            
            # Phase 3: LLM generation with context
            response = self._generate_response(
                question,
                memory_results,
                graph_analysis,
                context
            )
            
            # Phase 4: Quality assessment
            quality = self._assess_quality(response, graph_analysis)
            
            # Create cycle result
            result = CycleResult(
                response=response['text'],
                reasoning_trace=response.get('reasoning', ''),
                memory_used=memory_results,
                spike_detected=graph_analysis.get('spike_detected', False),
                graph_metrics=graph_analysis.get('metrics', {}),
                reasoning_quality=quality,
                convergence_score=self._calculate_convergence(context),
                has_spike=graph_analysis.get('spike_detected', False)
            )
            
            if verbose:
                logger.info(
                    f"Cycle complete - Quality: {quality:.3f}, "
                    f"Spike: {result.spike_detected}"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Reasoning cycle failed: {e}")
            return self._error_result(str(e))
    
    def check_convergence(
        self,
        results: List[CycleResult],
        question: str
    ) -> bool:
        """
        Check if reasoning has converged.
        
        Args:
            results: Recent cycle results
            question: Original question
            
        Returns:
            True if converged
        """
        if len(results) < 2:
            return False
        
        # Check quality convergence
        recent_quality = [r.reasoning_quality for r in results[-3:]]
        if all(q > self._min_quality_threshold for q in recent_quality):
            return True
        
        # Check answer stability
        recent_answers = [r.response for r in results[-3:]]
        if len(set(recent_answers)) == 1:
            return True
        
        # Check spike detection
        if any(r.spike_detected for r in results[-2:]):
            return True
        
        return False
    
    def _retrieve_relevant_memory(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memory using graph-based search."""
        # This would use the memory search interface
        # Implementation depends on specific memory architecture
        return context.get('memory_results', [])
    
    def _analyze_graph_state(
        self,
        memory_results: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze graph state and detect spikes."""
        # Use graph reasoner
        documents = [
            {
                'text': m.get('text', ''),
                'embedding': m.get('embedding'),
                'metadata': m.get('metadata', {})
            }
            for m in memory_results
        ]
        
        return self.graph_reasoner.analyze_documents(documents, context)
    
    def _generate_response(
        self,
        question: str,
        memory_results: List[Dict[str, Any]],
        graph_analysis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate response using LLM."""
        # Use LLM interface
        input_data = {
            'question': question,
            'context': memory_results,
            'graph_analysis': graph_analysis,
            'reasoning_quality': graph_analysis.get('reasoning_quality', 0.5)
        }
        
        return self.llm_interface.generate_response(input_data)
    
    def _assess_quality(
        self,
        response: Dict[str, Any],
        graph_analysis: Dict[str, Any]
    ) -> float:
        """Assess response quality."""
        # Combine multiple factors
        base_quality = graph_analysis.get('reasoning_quality', 0.5)
        response_length = len(response.get('text', ''))
        
        # Simple heuristic for now
        if response_length < 10:
            return base_quality * 0.5
        elif response_length > 1000:
            return base_quality * 0.9
        else:
            return base_quality
    
    def _calculate_convergence(self, context: Dict[str, Any]) -> float:
        """Calculate convergence score."""
        # Placeholder implementation
        return context.get('convergence_score', 0.0)
    
    def _error_result(self, error_msg: str) -> CycleResult:
        """Create error result."""
        return CycleResult(
            response=f"Error: {error_msg}",
            reasoning_trace="",
            memory_used=[],
            spike_detected=False,
            graph_metrics={},
            reasoning_quality=0.0,
            convergence_score=0.0,
            has_spike=False
        )