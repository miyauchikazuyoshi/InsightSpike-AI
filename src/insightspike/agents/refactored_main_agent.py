"""
Refactored Main Agent
====================

Clean implementation of MainAgent using dependency injection and separated responsibilities.
"""

import logging
from typing import Dict, Any, List, Optional, Union

from ..interfaces import IAgent
from ..core import (
    CycleResult,
    ReasoningEngine,
    MemoryController,
    ResponseGenerator
)
from ..config import InsightSpikeConfig
from ..di import DIContainer

logger = logging.getLogger(__name__)


class RefactoredMainAgent:
    """
    Refactored MainAgent with clean architecture.
    
    This implementation:
    - Uses dependency injection for all components
    - Separates concerns into specialized controllers
    - Follows SOLID principles
    - Provides clean interfaces
    """
    
    def __init__(
        self,
        config: InsightSpikeConfig,
        container: Optional[DIContainer] = None
    ):
        """
        Initialize agent with configuration and DI container.
        
        Args:
            config: Agent configuration
            container: Dependency injection container
        """
        self.config = config
        self.container = container or self._create_default_container()
        
        # Resolve core components
        self.reasoning_engine = self.container.resolve(ReasoningEngine)
        self.memory_controller = self.container.resolve(MemoryController)
        self.response_generator = self.container.resolve(ResponseGenerator)
        
        # State
        self._conversation_history: List[Dict[str, str]] = []
        self._current_context: Dict[str, Any] = {}
        
        logger.info("RefactoredMainAgent initialized with DI container")
    
    def process_question(
        self,
        question: str,
        max_cycles: int = 10,
        verbose: bool = False
    ) -> CycleResult:
        """
        Process a question through reasoning cycles.
        
        Args:
            question: Question to process
            max_cycles: Maximum reasoning cycles
            verbose: Whether to log details
            
        Returns:
            Final CycleResult
        """
        logger.info(f"Processing question: {question}")
        
        # Initialize context
        self._update_context(question)
        
        # Execute reasoning cycles
        cycle_results = []
        
        for cycle_num in range(max_cycles):
            # Execute single cycle
            result = self.reasoning_engine.execute_cycle(
                question,
                self._current_context,
                verbose
            )
            
            cycle_results.append(result)
            
            # Update context with result
            self._update_context_with_result(result)
            
            # Check convergence
            if self.reasoning_engine.check_convergence(cycle_results, question):
                logger.info(f"Converged after {cycle_num + 1} cycles")
                break
            
            # Log intermediate if verbose
            if verbose:
                intermediate = self.response_generator.format_intermediate_response(
                    result, cycle_num + 1
                )
                print(intermediate)
        
        # Generate final response
        final_response = self._finalize_response(cycle_results, question)
        
        # Update conversation history
        self._conversation_history.append({
            "question": question,
            "response": final_response.response,
            "cycles": len(cycle_results)
        })
        
        return final_response
    
    def add_knowledge(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add knowledge to memory.
        
        Args:
            text: Knowledge text
            metadata: Optional metadata
        """
        episode = self.memory_controller.add_knowledge(text, metadata)
        logger.info(f"Added knowledge: {episode.id}")
    
    def clear_memory(self, preserve_high_value: bool = False) -> None:
        """
        Clear agent memory.
        
        Args:
            preserve_high_value: Whether to keep high-value episodes
        """
        self.memory_controller.clear_memory(preserve_high_value)
        self._conversation_history.clear()
        self._current_context.clear()
        logger.info("Memory cleared")
    
    def get_memory_state(self) -> Dict[str, Any]:
        """
        Get current memory state.
        
        Returns:
            Memory state information
        """
        return self.memory_controller.get_memory_state()
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history.
        
        Returns:
            List of conversation entries
        """
        return self._conversation_history.copy()
    
    def save_to_disk(self, path: str) -> None:
        """
        Save agent state to disk.
        
        Args:
            path: Save path
        """
        # This would use the datastore through memory controller
        logger.info(f"Saving agent state to {path}")
        # Implementation depends on datastore interface
    
    def load_from_disk(self, path: str) -> None:
        """
        Load agent state from disk.
        
        Args:
            path: Load path
        """
        # This would use the datastore through memory controller
        logger.info(f"Loading agent state from {path}")
        self.memory_controller.load_from_datastore()
    
    def _create_default_container(self) -> DIContainer:
        """Create default DI container with all dependencies."""
        from ..di.container import DIContainer
        from ..di.providers import (
            DataStoreProvider,
            EmbedderProvider,
            LLMProviderFactory,
            GraphBuilderProvider,
            MemoryManagerProvider
        )
        from ..interfaces import (
            IDataStore,
            IEmbedder,
            ILLMProvider,
            IGraphBuilder,
            IMemoryManager,
            IMemorySearch
        )
        from ..implementations.layers import L3GraphReasoner, L4LLMInterface
        from ..implementations.memory.graph_memory_search import GraphMemorySearch
        
        container = DIContainer()
        
        # Register configuration
        container.instance(InsightSpikeConfig, self.config)
        
        # Register providers
        container.singleton(IDataStore, DataStoreProvider.create)
        container.singleton(IEmbedder, EmbedderProvider.create)
        container.singleton(ILLMProvider, LLMProviderFactory.create)
        container.singleton(IGraphBuilder, GraphBuilderProvider.create)
        container.singleton(IMemoryManager, MemoryManagerProvider.create)
        
        # Register memory search
        container.singleton(IMemorySearch, lambda c: GraphMemorySearch(
            c.resolve(IEmbedder),
            c.resolve(IGraphBuilder)
        ))
        
        # Register layers
        container.singleton(L3GraphReasoner, lambda c: L3GraphReasoner(
            c.resolve(InsightSpikeConfig)
        ))
        
        container.singleton(L4LLMInterface, lambda c: L4LLMInterface(
            c.resolve(ILLMProvider),
            c.resolve(InsightSpikeConfig)
        ))
        
        # Register core components
        container.singleton(ReasoningEngine, lambda c: ReasoningEngine(
            c.resolve(ILLMProvider),
            c.resolve(IMemorySearch),
            c.resolve(L3GraphReasoner),
            c.resolve(L4LLMInterface)
        ))
        
        container.singleton(MemoryController, lambda c: MemoryController(
            c.resolve(IMemoryManager),
            c.resolve(IDataStore),
            c.resolve(IEmbedder),
            c.resolve(IGraphBuilder)
        ))
        
        container.singleton(ResponseGenerator, lambda c: ResponseGenerator(
            c.resolve(ILLMProvider),
            self.config.response.style if hasattr(self.config, 'response') else 'concise'
        ))
        
        return container
    
    def _update_context(self, question: str) -> None:
        """Update current context with question."""
        # Retrieve relevant memory
        memory_results = self.memory_controller.retrieve_relevant(
            question,
            k=self.config.memory.search_k if hasattr(self.config.memory, 'search_k') else 10
        )
        
        self._current_context.update({
            "question": question,
            "memory_results": memory_results,
            "conversation_history": self._conversation_history[-5:],  # Last 5 entries
            "memory_state": self.memory_controller.get_memory_state()
        })
    
    def _update_context_with_result(self, result: CycleResult) -> None:
        """Update context with cycle result."""
        self._current_context["last_result"] = result
        self._current_context["convergence_score"] = result.convergence_score
        
        # Update episode rewards based on quality
        if result.reasoning_quality > 0.7:
            for memory_item in result.memory_used:
                if isinstance(memory_item, dict) and 'episode' in memory_item:
                    episode = memory_item['episode']
                    reward = 0.1 * result.reasoning_quality
                    self.memory_controller.update_episode_reward(episode.id, reward)
    
    def _finalize_response(
        self,
        cycle_results: List[CycleResult],
        question: str
    ) -> CycleResult:
        """Generate final response from all cycles."""
        if not cycle_results:
            # No cycles executed
            return CycleResult(
                response="Unable to process question",
                reasoning_trace="",
                memory_used=[],
                spike_detected=False,
                graph_metrics={},
                reasoning_quality=0.0,
                convergence_score=0.0,
                has_spike=False
            )
        
        # Use response generator for final formatting
        final_text = self.response_generator.generate_final_response(
            cycle_results,
            question,
            self._current_context
        )
        
        # Get best result for other fields
        best_result = max(cycle_results, key=lambda r: r.reasoning_quality)
        
        # Create final result with formatted response
        return CycleResult(
            response=final_text,
            reasoning_trace=best_result.reasoning_trace,
            memory_used=best_result.memory_used,
            spike_detected=any(r.spike_detected for r in cycle_results),
            graph_metrics=best_result.graph_metrics,
            reasoning_quality=best_result.reasoning_quality,
            convergence_score=best_result.convergence_score,
            has_spike=any(r.has_spike for r in cycle_results)
        )