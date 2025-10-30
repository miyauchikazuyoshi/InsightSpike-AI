"""
Refactored MainAgent with DataStore Integration
===============================================

This is a refactored version of MainAgent that properly uses DataStore
to prevent memory explosion while maintaining backward compatibility.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ...core.base import AgentInterface
from ...core.base.datastore import DataStore
from ...core.episode import Episode
from ...core.types import CycleResult
from ...config import InsightSpikeConfig
from ...monitoring.memory_monitor import get_memory_monitor
from ..layers import L1ErrorMonitor, L3GraphReasoner, L4LLMInterface
from ..layers.cached_memory_manager import CachedMemoryManager
from ..llm_providers import get_llm_provider

logger = logging.getLogger(__name__)


class MainAgentRefactored(AgentInterface):
    """
    Refactored MainAgent that uses DataStore properly.
    
    Key improvements:
    - Uses CachedMemoryManager instead of L2MemoryManager
    - All data operations go through DataStore
    - Built-in memory monitoring
    - Maintains API compatibility
    """
    
    def __init__(
        self,
        config: Union[InsightSpikeConfig, Dict[str, Any]] = None,
        datastore: Optional[DataStore] = None
    ):
        """
        Initialize refactored MainAgent.
        
        Args:
            config: Configuration
            datastore: DataStore instance (required)
        """
        if config is None:
            raise ValueError("Config must be provided to MainAgent")
            
        if datastore is None:
            raise ValueError("DataStore must be provided to prevent memory explosion")
            
        # Configuration
        if isinstance(config, InsightSpikeConfig):
            self.config = config
            self.is_pydantic_config = True
        else:
            self.config = config
            self.is_pydantic_config = False
            
        # Core components
        self.datastore = datastore
        
        # Initialize layers
        self.l1_error_monitor = L1ErrorMonitor(self.config)
        
        # Use CachedMemoryManager instead of L2MemoryManager
        self.l2_memory = CachedMemoryManager(
            datastore=self.datastore,
            cache_size=100  # Small cache to prevent memory explosion
        )
        
        self.l3_graph = self._init_graph_reasoner()
        self.l4_llm = get_llm_provider(self.config, safe_mode=False)
        
        # Memory monitoring
        self.memory_monitor = get_memory_monitor()
        self._setup_memory_monitoring()
        
        # State tracking
        self.cycle_count = 0
        self.previous_state = {}
        self.reasoning_history = []
        
        logger.info("MainAgentRefactored initialized with DataStore backend")
        
    def _init_graph_reasoner(self) -> Optional[L3GraphReasoner]:
        """Initialize graph reasoner if available"""
        try:
            return L3GraphReasoner(self.config)
        except Exception as e:
            logger.warning(f"Graph reasoner not available: {e}")
            return None
            
    def _setup_memory_monitoring(self):
        """Setup memory monitoring with automatic cleanup"""
        def on_warning(snapshot):
            logger.warning(
                f"Memory warning: {snapshot.memory_mb:.1f} MB. "
                f"Clearing cache..."
            )
            self.l2_memory.clear_cache()
            
        def on_critical(snapshot):
            logger.critical(
                f"Memory critical: {snapshot.memory_mb:.1f} MB. "
                f"Emergency cleanup!"
            )
            # Clear all caches
            self.l2_memory.clear_cache()
            self.reasoning_history.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
        self.memory_monitor.add_warning_callback(on_warning)
        self.memory_monitor.add_critical_callback(on_critical)
        
    def add_knowledge(
        self,
        text: str,
        c_value: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add new knowledge to the system.
        
        This now properly uses DataStore instead of keeping everything in memory.
        """
        # Monitor memory
        self.memory_monitor.check_memory(
            episode_count=self._estimate_episode_count(),
            cache_size=len(self.l2_memory.cache)
        )
        
        # Add via cached memory manager (which uses DataStore)
        episode_id = self.l2_memory.add_episode(
            text=text,
            c_value=c_value,
            metadata=metadata
        )
        
        # Update graph if available
        if self.l3_graph:
            self._update_graph()
            
        logger.info(f"Added knowledge with ID: {episode_id}")
        return episode_id  # Return string ID, not index
        
    def process_question(
        self,
        question: str,
        max_cycles: int = 3,
        debug: bool = False
    ) -> Union[CycleResult, Dict[str, Any]]:
        """
        Process a question using minimal memory footprint.
        """
        start_time = time.time()
        self.cycle_count += 1
        
        # Monitor memory
        self.memory_monitor.check_memory(
            episode_count=self._estimate_episode_count(),
            cache_size=len(self.l2_memory.cache)
        )
        
        # L1: Analyze uncertainty
        uncertainty_analysis = self.l1_error_monitor.analyze_uncertainty(
            question,
            self.previous_state
        )
        
        # Search for relevant episodes (via DataStore)
        relevant_episodes = self.l2_memory.search_episodes(
            query=question,
            top_k=10
        )
        
        # L3: Graph reasoning
        graph_insights = None
        if self.l3_graph and relevant_episodes:
            try:
                # Get embeddings for graph reasoning
                embeddings = np.array([ep.vec for ep, _ in relevant_episodes])
                graph = self.l3_graph.reason(embeddings)
                graph_insights = {"connections": len(graph.edges) if graph else 0}
            except Exception as e:
                logger.error(f"Graph reasoning failed: {e}")
                
        # L4: Generate response
        context = self._build_context(relevant_episodes, graph_insights)
        response = self.l4_llm.generate(
            prompt=question,
            context=context,
            temperature=0.7,
            max_tokens=500
        )
        
        # Detect insight spike
        has_spike = self._detect_spike(
            question,
            response,
            uncertainty_analysis
        )
        
        # Update state
        self.previous_state = {
            'question': question,
            'response': response,
            'has_spike': has_spike,
            'uncertainty': uncertainty_analysis.get('uncertainty', 0.5)
        }
        
        # Add to reasoning history (with size limit)
        self.reasoning_history.append(self.previous_state)
        if len(self.reasoning_history) > 10:
            self.reasoning_history.pop(0)
            
        # Performance metrics
        processing_time = time.time() - start_time
        
        # Return result
        result = CycleResult(
            response=response,
            has_spike=has_spike,
            reasoning_quality=self._calculate_quality(
                relevant_episodes,
                uncertainty_analysis
            ),
            total_cycles=self.cycle_count,
            metadata={
                'processing_time': processing_time,
                'episodes_retrieved': len(relevant_episodes),
                'memory_mb': self.memory_monitor.get_memory_usage_mb(),
                'cache_stats': self.l2_memory.get_stats()
            }
        )
        
        # For backward compatibility
        if debug:
            return result.__dict__
        return result
        
    def _build_context(
        self,
        episodes: List[Tuple[Episode, float]],
        graph_insights: Optional[Dict[str, Any]]
    ) -> str:
        """Build context string from episodes and insights"""
        context_parts = []
        
        # Add top episodes
        for episode, similarity in episodes[:5]:
            context_parts.append(
                f"[Relevance: {similarity:.2f}] {episode.text}"
            )
            
        # Add graph insights
        if graph_insights:
            context_parts.append(
                f"\nGraph connections: {graph_insights.get('connections', 0)}"
            )
            
        return "\n".join(context_parts)
        
    def _detect_spike(
        self,
        question: str,
        response: str,
        uncertainty_analysis: Dict[str, Any]
    ) -> bool:
        """Detect if this represents an insight spike"""
        uncertainty = uncertainty_analysis.get('uncertainty', 0.5)
        
        # High uncertainty + novel response = potential spike
        if uncertainty > 0.7:
            # Check if response is significantly different from history
            if self.reasoning_history:
                # Simple novelty check
                recent_responses = [h['response'] for h in self.reasoning_history[-3:]]
                if all(response[:50] != r[:50] for r in recent_responses):
                    return True
                    
        return False
        
    def _calculate_quality(
        self,
        episodes: List[Tuple[Episode, float]],
        uncertainty_analysis: Dict[str, Any]
    ) -> float:
        """Calculate reasoning quality score"""
        if not episodes:
            return 0.3
            
        # Average relevance
        avg_relevance = np.mean([sim for _, sim in episodes])
        
        # Confidence from uncertainty
        uncertainty = uncertainty_analysis.get('uncertainty', 0.5)
        confidence = 1.0 - uncertainty
        
        # Combined quality
        quality = (avg_relevance * 0.7 + confidence * 0.3)
        return float(np.clip(quality, 0.0, 1.0))
        
    def _update_graph(self):
        """Update graph structure in DataStore"""
        if not self.l3_graph:
            return
            
        try:
            # Get recent episodes for graph building
            recent_episodes = self.l2_memory.get_recent_episodes(limit=100)
            if not recent_episodes:
                return
                
            # Build graph from embeddings
            embeddings = np.array([ep.vec for ep in recent_episodes])
            
            # This should be updated to use a proper graph builder
            # For now, we'll skip the actual graph update
            logger.debug("Graph update skipped in refactored version")
            
        except Exception as e:
            logger.error(f"Graph update failed: {e}")
            
    def _estimate_episode_count(self) -> int:
        """Estimate total episodes in system"""
        try:
            stats = self.l2_memory.get_stats()
            return stats.get('estimated_episodes', 0)
        except Exception:
            return 0
            
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        memory_stats = self.memory_monitor.get_stats()
        cache_stats = self.l2_memory.get_stats()
        
        return {
            'cycle_count': self.cycle_count,
            'memory_mb': memory_stats['current_mb'],
            'memory_trend': memory_stats['trend'],
            'cache_size': cache_stats['cache_size'],
            'cache_hit_rate': cache_stats.get('cache_hit_rate', 0),
            'estimated_episodes': cache_stats.get('estimated_episodes', 0),
            'datastore_type': type(self.datastore).__name__
        }
        
    def save(self, filepath: Optional[str] = None) -> bool:
        """Save agent state via DataStore"""
        logger.info("Save handled automatically by DataStore")
        return True
        
    def load(self, filepath: Optional[str] = None) -> bool:
        """Load agent state via DataStore"""
        logger.info("Load handled automatically by DataStore")
        return True
        
    # Backward compatibility methods
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Backward compatibility for semantic search"""
        results = self.l2_memory.search_episodes(query, top_k)
        
        # Convert to expected format
        formatted_results = []
        for i, (episode, similarity) in enumerate(results):
            formatted_results.append({
                'index': i,
                'text': episode.text,
                'similarity': similarity,
                'c_value': episode.c
            })
            
        return formatted_results