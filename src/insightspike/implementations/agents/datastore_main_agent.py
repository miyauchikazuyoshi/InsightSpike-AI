"""
DataStore-based MainAgent Implementation
========================================

A new MainAgent that uses only DataStore for memory management,
eliminating in-memory episode storage to prevent memory explosion.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np

from ...core.base import AgentInterface
from ...core.base.datastore import DataStore
from ...core.episode import Episode
from ...core.types import CycleResult
from ...config import InsightSpikeConfig, get_config
from ...detection.spike_detector import SpikeDetector
from ...monitoring.memory_monitor import MemoryMonitor, get_memory_monitor
from ..layers import L1ErrorMonitor, L3GraphReasoner, L4LLMInterface
from ..llm_providers import get_llm_provider
from ...processing.embedder import EmbeddingManager

logger = logging.getLogger(__name__)


@dataclass
class QueryCache:
    """Simple cache for recent queries"""
    query: str
    embedding: np.ndarray
    response: str
    timestamp: float
    has_spike: bool


class DataStoreMainAgent(AgentInterface):
    """
    MainAgent implementation that uses DataStore exclusively for memory management.
    
    Key differences from original MainAgent:
    - No in-memory episode list
    - All data operations through DataStore
    - Intelligent caching for performance
    - Built-in memory monitoring
    """
    
    def __init__(
        self, 
        config: Union[InsightSpikeConfig, Dict[str, Any]],
        datastore: DataStore,
        memory_monitor: Optional[MemoryMonitor] = None
    ):
        """
        Initialize DataStore-based MainAgent.
        
        Args:
            config: Configuration (InsightSpikeConfig or legacy format)
            datastore: DataStore instance for all data operations
            memory_monitor: Optional memory monitor instance
        """
        if not datastore:
            raise ValueError("DataStore is required for DataStoreMainAgent")
            
        # Configuration
        if isinstance(config, InsightSpikeConfig):
            self.config = config
            self.is_pydantic_config = True
        else:
            self.config = config
            self.is_pydantic_config = False
            
        # Core components
        self.datastore = datastore
        self.memory_monitor = memory_monitor or get_memory_monitor()
        
        # Initialize layers
        self.l1_error_monitor = L1ErrorMonitor(config)
        self.l3_graph = L3GraphReasoner(config) if self._is_graph_available() else None
        self.l4_llm = get_llm_provider(config, safe_mode=False)
        
        # Tools
        self.embedder = EmbeddingManager()
        self.spike_detector = SpikeDetector()
        
        # Caching for performance
        self.query_cache: List[QueryCache] = []
        self.max_cache_size = 100
        self.cache_ttl = 3600  # 1 hour
        
        # State tracking
        self.cycle_count = 0
        self.previous_state = {}
        
        # Memory monitoring callbacks
        self._setup_memory_callbacks()
        
        logger.info("DataStoreMainAgent initialized with DataStore backend")
        
    def _setup_memory_callbacks(self):
        """Setup memory monitoring callbacks"""
        def on_warning(snapshot):
            logger.warning(f"Memory warning: {snapshot.memory_mb:.1f} MB used")
            # Trigger cache cleanup
            self._cleanup_cache(aggressive=False)
            
        def on_critical(snapshot):
            logger.critical(f"Memory critical: {snapshot.memory_mb:.1f} MB used")
            # Aggressive cleanup
            self._cleanup_cache(aggressive=True)
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
    ) -> str:
        """
        Add new knowledge to the system via DataStore.
        
        Args:
            text: Knowledge text
            c_value: Initial importance value
            metadata: Optional metadata
            
        Returns:
            Episode ID
        """
        # Check memory before adding
        self.memory_monitor.check_memory(
            episode_count=self._get_episode_count(),
            cache_size=len(self.query_cache)
        )
        
        # Create episode
        embedding = self.embedder.embed_text(text)
        episode = Episode(
            text=text,
            vec=embedding,
            c=c_value,
            metadata=metadata or {}
        )
        
        # Add to DataStore
        episode_id = self.datastore.add_episode(episode)
        
        # Update graph if available
        if self.l3_graph and hasattr(self.datastore, 'update_graph'):
            try:
                # Get recent episodes for graph update
                recent_episodes = self.datastore.get_recent_episodes(limit=100)
                embeddings = np.array([ep.vec for ep in recent_episodes])
                
                # Build graph
                graph = self.l3_graph.build_graph(embeddings)
                
                # Save graph to DataStore
                self.datastore.save_graph(graph)
                
            except Exception as e:
                logger.error(f"Failed to update graph: {e}")
                
        logger.info(f"Added knowledge: '{text[:50]}...' with ID: {episode_id}")
        return episode_id
        
    def process_question(
        self, 
        question: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> CycleResult:
        """
        Process a question using DataStore for memory retrieval.
        
        Args:
            question: Input question
            temperature: LLM temperature
            max_tokens: Max response tokens
            
        Returns:
            CycleResult with response and metadata
        """
        start_time = time.time()
        self.cycle_count += 1
        
        # Check cache first
        cached = self._check_cache(question)
        if cached:
            logger.info("Returning cached response")
            return CycleResult(
                response=cached.response,
                has_spike=cached.has_spike,
                reasoning_quality=0.9,
                total_cycles=self.cycle_count,
                metadata={'cached': True}
            )
            
        # Memory check
        self.memory_monitor.check_memory(
            episode_count=self._get_episode_count(),
            cache_size=len(self.query_cache)
        )
        
        # L1: Error monitoring
        uncertainty_analysis = self.l1_error_monitor.analyze_uncertainty(
            question, 
            self.previous_state
        )
        
        # Get embedding for search
        query_embedding = self.embedder.embed_text(question)
        
        # Search DataStore for relevant episodes
        search_results = self.datastore.search_episodes_by_vector(
            query_embedding,
            top_k=10
        )
        
        # Extract context from results
        context_episodes = []
        for episode_id, similarity in search_results:
            episode = self.datastore.get_episode(episode_id)
            if episode:
                context_episodes.append((episode, similarity))
                
        # L3: Graph reasoning (if available)
        graph_context = None
        if self.l3_graph and hasattr(self.datastore, 'get_graph'):
            try:
                graph = self.datastore.get_graph()
                if graph:
                    graph_context = self.l3_graph.reason_with_graph(
                        query_embedding,
                        graph,
                        context_episodes
                    )
            except Exception as e:
                logger.error(f"Graph reasoning failed: {e}")
                
        # L4: Generate response
        response = self._generate_response(
            question,
            context_episodes,
            graph_context,
            temperature,
            max_tokens
        )
        
        # Spike detection
        has_spike = self._detect_spike(
            question,
            response,
            context_episodes,
            uncertainty_analysis
        )
        
        # Update state
        self.previous_state = {
            'question': question,
            'response': response,
            'has_spike': has_spike,
            'uncertainty': uncertainty_analysis.get('uncertainty', 0.5)
        }
        
        # Cache the result
        self._add_to_cache(question, query_embedding, response, has_spike)
        
        # Performance tracking
        processing_time = time.time() - start_time
        
        return CycleResult(
            response=response,
            has_spike=has_spike,
            reasoning_quality=self._calculate_quality(
                context_episodes,
                uncertainty_analysis
            ),
            total_cycles=self.cycle_count,
            metadata={
                'processing_time': processing_time,
                'context_size': len(context_episodes),
                'uncertainty': uncertainty_analysis.get('uncertainty', 0.5),
                'memory_mb': self.memory_monitor.get_memory_usage_mb()
            }
        )
        
    def _generate_response(
        self,
        question: str,
        context_episodes: List[Tuple[Episode, float]],
        graph_context: Optional[Dict[str, Any]],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate response using LLM with context"""
        # Build context string
        context_parts = []
        for episode, similarity in context_episodes[:5]:  # Top 5
            context_parts.append(f"[{similarity:.2f}] {episode.text}")
            
        context = "\n".join(context_parts)
        
        # Add graph insights if available
        if graph_context:
            context += f"\n\nGraph insights: {graph_context.get('summary', '')}"
            
        # Generate response
        try:
            response = self.l4_llm.generate(
                prompt=question,
                context=context,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "I encountered an error generating a response."
            
    def _detect_spike(
        self,
        question: str,
        response: str,
        context_episodes: List[Tuple[Episode, float]],
        uncertainty_analysis: Dict[str, Any]
    ) -> bool:
        """Detect if the current cycle represents an insight spike"""
        # High uncertainty + novel response pattern
        uncertainty = uncertainty_analysis.get('uncertainty', 0.5)
        
        # Check if response is significantly different from context
        if context_episodes:
            avg_similarity = np.mean([sim for _, sim in context_episodes])
            if avg_similarity < 0.5 and uncertainty > 0.7:
                return True
                
        # Use spike detector for additional checks
        if hasattr(self.spike_detector, 'detect'):
            return self.spike_detector.detect(
                question=question,
                response=response,
                uncertainty=uncertainty
            )
            
        return False
        
    def _calculate_quality(
        self,
        context_episodes: List[Tuple[Episode, float]],
        uncertainty_analysis: Dict[str, Any]
    ) -> float:
        """Calculate reasoning quality score"""
        if not context_episodes:
            return 0.3
            
        # Average similarity of retrieved context
        avg_similarity = np.mean([sim for _, sim in context_episodes])
        
        # Inverse uncertainty contribution
        uncertainty = uncertainty_analysis.get('uncertainty', 0.5)
        confidence = 1.0 - uncertainty
        
        # Combined quality
        quality = (avg_similarity * 0.7 + confidence * 0.3)
        return float(np.clip(quality, 0.0, 1.0))
        
    def _check_cache(self, query: str) -> Optional[QueryCache]:
        """Check if query is in cache"""
        current_time = time.time()
        
        for cached in self.query_cache:
            if cached.query == query:
                # Check TTL
                if (current_time - cached.timestamp) < self.cache_ttl:
                    return cached
                    
        return None
        
    def _add_to_cache(
        self, 
        query: str, 
        embedding: np.ndarray,
        response: str, 
        has_spike: bool
    ):
        """Add query result to cache"""
        # Clean old entries if needed
        if len(self.query_cache) >= self.max_cache_size:
            self._cleanup_cache(aggressive=False)
            
        self.query_cache.append(QueryCache(
            query=query,
            embedding=embedding,
            response=response,
            timestamp=time.time(),
            has_spike=has_spike
        ))
        
    def _cleanup_cache(self, aggressive: bool = False):
        """Clean up cache to free memory"""
        current_time = time.time()
        
        if aggressive:
            # Keep only 10% newest entries
            keep_count = max(1, len(self.query_cache) // 10)
            self.query_cache = sorted(
                self.query_cache, 
                key=lambda x: x.timestamp, 
                reverse=True
            )[:keep_count]
        else:
            # Remove expired entries
            self.query_cache = [
                c for c in self.query_cache
                if (current_time - c.timestamp) < self.cache_ttl
            ]
            
    def _get_episode_count(self) -> int:
        """Get approximate episode count from DataStore"""
        try:
            # This should be a fast metadata query, not loading all episodes
            if hasattr(self.datastore, 'get_episode_count'):
                return self.datastore.get_episode_count()
            else:
                # Fallback: try to get recent episodes count
                recent = self.datastore.get_recent_episodes(limit=1)
                # Assume there are many more
                return len(recent) * 1000  # Rough estimate
        except Exception:
            return 0
            
    def _is_graph_available(self) -> bool:
        """Check if graph reasoning is available"""
        try:
            from ..layers import L3GraphReasoner
            return True
        except ImportError:
            return False
            
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        memory_stats = self.memory_monitor.get_stats()
        
        return {
            'cycle_count': self.cycle_count,
            'cache_size': len(self.query_cache),
            'memory_mb': memory_stats['current_mb'],
            'memory_trend': memory_stats['trend'],
            'warnings': memory_stats['warnings'],
            'datastore_type': type(self.datastore).__name__,
            'episode_count': self._get_episode_count()
        }
        
    def save(self) -> bool:
        """Save is handled by DataStore automatically"""
        logger.info("DataStoreMainAgent uses automatic persistence via DataStore")
        return True
        
    def load(self) -> bool:
        """Load is handled by DataStore automatically"""
        logger.info("DataStoreMainAgent uses automatic loading via DataStore")
        return True