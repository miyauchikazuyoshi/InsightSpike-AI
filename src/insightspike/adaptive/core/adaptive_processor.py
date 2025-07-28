"""
Adaptive Processor - Main coordinator for adaptive exploration
"""

import logging
import time
from typing import Dict, List, Any, Optional

from ..core.interfaces import (
    ExplorationParams,
    ExplorationResult,
    ExplorationStrategy,
    TopKCalculator,
    PatternLearner
)
from .exploration_loop import ExplorationLoop

logger = logging.getLogger(__name__)


class AdaptiveProcessor:
    """
    Main adaptive processing coordinator.
    
    Orchestrates the exploration loop with strategy-based parameter adjustment
    until a spike is detected or max attempts reached.
    """
    
    def __init__(
        self,
        exploration_loop: ExplorationLoop,
        strategy: ExplorationStrategy,
        topk_calculator: TopKCalculator,
        l4_llm,
        pattern_learner: Optional[PatternLearner] = None,
        max_attempts: int = 5
    ):
        """
        Initialize adaptive processor.
        
        Args:
            exploration_loop: The L1-L2-L3 exploration loop
            strategy: Exploration strategy for parameter adjustment
            topk_calculator: Calculator for adaptive topK values
            l4_llm: Layer 4 LLM interface (called only after spike)
            pattern_learner: Optional pattern learning component
            max_attempts: Maximum exploration attempts
        """
        self.exploration_loop = exploration_loop
        self.strategy = strategy
        self.topk_calculator = topk_calculator
        self.l4_llm = l4_llm
        self.pattern_learner = pattern_learner
        self.max_attempts = max_attempts
        
    def process(
        self, 
        question: str,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Process question with adaptive exploration.
        
        Args:
            question: The query to process
            verbose: Whether to log detailed progress
            
        Returns:
            Dict containing response and processing metadata
        """
        start_time = time.time()
        
        if verbose:
            logger.info(f"🔄 Starting adaptive processing for: {question[:50]}...")
        
        # Check learned patterns first
        if self.pattern_learner:
            suggested_params = self.pattern_learner.suggest_params(question)
            if suggested_params:
                if verbose:
                    logger.info("📚 Using learned parameters for first attempt")
                    
                # Try learned parameters
                result = self.exploration_loop.explore_once(question, suggested_params)
                if result.spike_detected:
                    return self._generate_response(
                        question, 
                        result, 
                        [result],
                        time.time() - start_time
                    )
        
        # Run adaptive exploration
        exploration_results = []
        params = self.strategy.get_initial_params()
        
        for attempt in range(self.max_attempts):
            # Update attempt number
            params.attempt_number = attempt + 1
            
            if verbose:
                logger.info(
                    f"\n--- Attempt {attempt + 1}/{self.max_attempts} ---\n"
                    f"Radius: {params.radius:.2f}, TopK: {params.topk_l2}"
                )
            
            # Single exploration attempt
            result = self.exploration_loop.explore_once(question, params)
            exploration_results.append(result)
            
            if verbose:
                logger.info(
                    f"Result: confidence={result.confidence:.2f}, "
                    f"spike={'YES' if result.spike_detected else 'NO'}, "
                    f"docs={len(result.retrieved_docs)}"
                )
            
            # Check for spike
            if result.spike_detected:
                if verbose:
                    logger.info(
                        f"✅ Spike detected after {attempt + 1} attempts! "
                        f"Calling LLM..."
                    )
                
                # Record successful pattern
                if self.pattern_learner:
                    exploration_path = [r.params for r in exploration_results]
                    self.pattern_learner.record_success(
                        question,
                        exploration_path,
                        result
                    )
                
                # Generate response with LLM
                return self._generate_response(
                    question,
                    result,
                    exploration_results,
                    time.time() - start_time
                )
            
            # Check if should continue
            if not self.strategy.should_continue(exploration_results):
                if verbose:
                    logger.info("🛑 Strategy decided to stop exploration")
                break
            
            # Adjust parameters for next attempt
            params = self.strategy.adjust_params(attempt + 1, result)
        
        # No spike found - use best result
        if verbose:
            logger.info(
                f"⚠️ No spike found after {len(exploration_results)} attempts. "
                f"Using best result..."
            )
            
        best_result = max(exploration_results, key=lambda r: r.confidence)
        return self._generate_response(
            question,
            best_result,
            exploration_results,
            time.time() - start_time
        )
    
    def _generate_response(
        self,
        question: str,
        final_result: ExplorationResult,
        all_results: List[ExplorationResult],
        processing_time: float
    ) -> Dict[str, Any]:
        """Generate final response using L4 LLM"""
        
        # Prepare context for LLM
        # Convert documents to proper format if needed
        retrieved_docs = final_result.retrieved_docs
        if retrieved_docs and isinstance(retrieved_docs[0], dict):
            # Extract text from document dicts
            doc_texts = [doc.get("text", str(doc)) for doc in retrieved_docs]
        else:
            doc_texts = retrieved_docs
            
        llm_context = {
            "retrieved_documents": doc_texts,
            "graph_analysis": final_result.graph_analysis,
            "reasoning_quality": final_result.confidence,
            "spike_detected": final_result.spike_detected,
            "exploration_attempts": len(all_results)
        }
        
        # Call LLM
        try:
            if hasattr(self.l4_llm, 'generate_response_detailed'):
                llm_result = self.l4_llm.generate_response_detailed(
                    llm_context,
                    question
                )
                response = llm_result.get("response", "")
                reasoning = llm_result.get("reasoning", "")
            else:
                response = self.l4_llm.generate_response(llm_context, question)
                reasoning = ""
                
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            response = f"Error generating response: {str(e)}"
            reasoning = ""
        
        # Build comprehensive result
        return {
            "response": response,
            "reasoning": reasoning,
            "spike_detected": final_result.spike_detected,
            "reasoning_quality": final_result.confidence,
            "retrieved_documents": final_result.retrieved_docs,
            "graph_analysis": final_result.graph_analysis,
            "adaptive_metadata": {
                "total_attempts": len(all_results),
                "processing_time": processing_time,
                "exploration_path": [r.params.to_dict() for r in all_results],
                "final_metrics": final_result.metrics,
                "api_calls": 1  # Only one LLM call!
            },
            "success": True
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = {
            "strategy_type": self.strategy.__class__.__name__,
            "max_attempts": self.max_attempts
        }
        
        if self.pattern_learner:
            stats["learning_stats"] = self.pattern_learner.get_statistics()
            
        return stats