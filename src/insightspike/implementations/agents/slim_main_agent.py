"""Slim Main Agent - Refactored for Minimal Complexity

This is a streamlined version of MainAgent with:
- Extracted diagnostic functionality
- Modular spike detection pipeline
- Clean fallback registry usage  
- Reduced configuration branching
- Clear separation of concerns

Target: 40% reduction in line count while preserving functionality.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ...core.base.datastore import DataStore
from ...core.episode import Episode
from ...detection.insight_registry import get_insight_registry
from ...spike_pipeline import SpikePipeline, SpikeDecisionMode
from ...fallback.registry import get_fallback_registry, FallbackReason, execute_fallback

logger = logging.getLogger(__name__)

# Layer imports - simplified
from ..layers.layer1_error_monitor import ErrorMonitor
from ..layers.layer2_compatibility import CompatibleL2MemoryManager as Memory
from ..memory.graph_memory_search import GraphMemorySearch

# Optional layer 3
try:
    from ..layers.layer3_graph_reasoner import L3GraphReasoner
    GRAPH_REASONER_AVAILABLE = True
except ImportError:
    GRAPH_REASONER_AVAILABLE = False
    L3GraphReasoner = None


@dataclass
class CycleResult:
    """Result from one reasoning cycle - simplified version."""
    question: str
    retrieved_documents: List[Dict[str, Any]]
    graph_analysis: Dict[str, Any]
    response: str
    reasoning_quality: float
    spike_detected: bool
    error_state: Dict[str, Any]
    has_spike: bool = False  # Legacy compatibility
    confidence: float = 0.0
    
    def __post_init__(self):
        self.has_spike = self.spike_detected
        self.confidence = self.reasoning_quality


class SlimMainAgent:
    """Streamlined main agent with modular architecture."""
    
    def __init__(self, config=None, datastore: Optional[DataStore] = None):
        """Initialize with clean configuration handling."""
        if config is None:
            raise ValueError("Config must be provided")
        
        # Normalize configuration once
        self.config = self._normalize_config(config)
        self.datastore = datastore
        
        # Initialize core layers
        self._init_layers()
        
        # Initialize pipeline components
        self._init_pipelines()
        
        # State tracking
        self.cycle_count = 0
        self.previous_state: Dict[str, Any] = {}
        self.reasoning_history: List[Dict[str, Any]] = []
        self._initialized = False
        
        logger.info("SlimMainAgent initialized successfully")
    
    def _normalize_config(self, raw_config):
        """Normalize configuration to consistent format."""
        try:
            from ...config.normalized import NormalizedConfig
            return NormalizedConfig.from_any(raw_config)
        except Exception as e:
            logger.warning(f"Config normalization failed: {e}, using raw config")
            return raw_config
    
    def _init_layers(self):
        """Initialize processing layers."""
        # Layer 1: Error monitoring
        self.l1_error_monitor = ErrorMonitor(self.config)
        
        # Layer 1b: Embeddings
        from ...processing.embedder import EmbeddingManager
        embedding_model = getattr(self.config, 'embedding_model', None)
        self.l1_embedder = EmbeddingManager(model_name=embedding_model, config=self.config)
        
        # Layer 2: Memory
        memory_dim = getattr(self.config, 'embedding_dim', 384)
        self.l2_memory = Memory(dim=memory_dim, config=self.config)
        
        # Layer 3: Graph reasoning (optional)
        self.l3_graph = L3GraphReasoner(self.config) if GRAPH_REASONER_AVAILABLE else None
        
        # Layer 4: LLM (lazy initialization)
        self.l4_llm = None
        
        logger.debug("Layers initialized")
    
    def _init_pipelines(self):
        """Initialize processing pipelines."""
        # Spike detection pipeline
        spike_mode = getattr(self.config, 'spike_decision_mode', 'weighted')
        mode_map = {
            'threshold': SpikeDecisionMode.THRESHOLD,
            'and': SpikeDecisionMode.AND,
            'or': SpikeDecisionMode.OR,
            'weighted': SpikeDecisionMode.WEIGHTED,
            'adaptive': SpikeDecisionMode.ADAPTIVE
        }
        
        self.spike_pipeline = SpikePipeline(
            config=self.config,
            decision_mode=mode_map.get(spike_mode, SpikeDecisionMode.WEIGHTED)
        )
        
        # Other pipeline components
        self.graph_memory_search = GraphMemorySearch(self.config)
        self.insight_registry = get_insight_registry()
        
        # Query recorder (if available)
        try:
            from ...query.query_recorder import QueryRecorder
            self.query_recorder = QueryRecorder(self.datastore)
        except Exception as e:
            logger.debug(f"QueryRecorder not available: {e}")
            self.query_recorder = None
        
        logger.debug("Pipelines initialized")
    
    def initialize(self) -> bool:
        """Initialize all components."""
        try:
            logger.info("Initializing SlimMainAgent components...")
            
            # Initialize LLM provider (lazy)
            if self.l4_llm is None:
                self.l4_llm = self._create_llm_provider()
            
            if hasattr(self.l4_llm, "initialize") and not getattr(self.l4_llm, 'initialized', False):
                if not self.l4_llm.initialize():
                    logger.error("Failed to initialize LLM provider")
                    return False
            
            # Load existing memory
            if not self.l2_memory.load():
                logger.info("No existing memory found, starting fresh")
            
            # Reset error monitor
            self.l1_error_monitor.reset()
            
            self._initialized = True
            logger.info("SlimMainAgent initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def _create_llm_provider(self):
        """Create LLM provider with fallback handling."""
        try:
            from ..layers.layer4_llm_interface import get_llm_provider
            return get_llm_provider(self.config, safe_mode=False)
        except Exception as e:
            logger.warning(f"Failed to create LLM provider: {e}, using fallback")
            
            # Simple fallback LLM
            class FallbackLLM:
                initialized = True
                def initialize(self): return True
                def generate(self, *args, **kwargs):
                    return {"text": "[Fallback response]", "raw": "[Fallback response]"}
            
            return FallbackLLM()
    
    def process_question(self, question: str, verbose: bool = False) -> CycleResult:
        """Process a question through the reasoning pipeline."""
        
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Agent initialization failed")
        
        start_time = time.time()
        
        try:
            # Execute reasoning cycle
            result = self._execute_cycle(question, verbose)
            
            # Save query if recorder available
            self._save_query(question, result, time.time() - start_time)
            
            # Update reasoning history
            self.reasoning_history.append({
                "question": question,
                "cycles": 1,
                "quality": result.reasoning_quality,
                "spike_detected": result.spike_detected
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Question processing failed: {e}")
            return self._create_error_result(question, str(e))
    
    def _execute_cycle(self, question: str, verbose: bool = False) -> CycleResult:
        """Execute one complete reasoning cycle."""
        self.cycle_count += 1
        
        try:
            # L1: Error monitoring and uncertainty
            error_state = self.l1_error_monitor.analyze_uncertainty(question, self.previous_state)
            
            # Check for L1 bypass (fast path)
            if self._should_bypass_l1(error_state):
                return self._execute_bypass_path(question, error_state)
            
            # L2: Memory search
            memory_results = self._search_memory(question)
            retrieved_docs = memory_results["documents"]
            
            # L3: Graph analysis and spike detection
            graph_analysis = self._analyze_graph(question, retrieved_docs, error_state)
            
            # Spike detection using new pipeline
            spike_result = self._detect_spike_via_pipeline(graph_analysis, retrieved_docs)
            
            # L4: Generate response
            llm_result = self._generate_response(question, retrieved_docs, graph_analysis, spike_result)
            
            # Create cycle result
            cycle_result = CycleResult(
                question=question,
                retrieved_documents=retrieved_docs,
                graph_analysis=graph_analysis,
                response=llm_result.get("text", ""),
                reasoning_quality=self._calculate_quality(spike_result, memory_results),
                spike_detected=spike_result.get("spike_detected", False),
                error_state=error_state
            )
            
            # Update state for next cycle
            self._update_state(cycle_result)
            
            return cycle_result
            
        except Exception as e:
            logger.error(f"Cycle execution failed: {e}")
            return self._create_error_result(question, str(e))
    
    def _should_bypass_l1(self, error_state: Dict[str, Any]) -> bool:
        """Check if L1 bypass should be used."""
        bypass_enabled = getattr(self.config, 'enable_layer1_bypass', False)
        uncertainty = error_state.get("uncertainty", 1.0)
        known_ratio = error_state.get("known_ratio", 0.0)
        
        return (bypass_enabled and 
                uncertainty < 0.2 and 
                known_ratio > 0.9 and
                len(error_state.get("known_elements", [])) > 0)
    
    def _execute_bypass_path(self, question: str, error_state: Dict[str, Any]) -> CycleResult:
        """Execute fast bypass path."""
        try:
            # Use known elements for fast response
            known_elements = error_state.get("known_elements", [])
            
            llm_result = self.l4_llm.generate(
                prompt=f"Answer concisely based on: {known_elements[:3]}\nQuestion: {question}",
                max_tokens=150
            )
            
            return CycleResult(
                question=question,
                retrieved_documents=[],
                graph_analysis={"bypass": True},
                response=llm_result.get("text", ""),
                reasoning_quality=0.8,  # Good quality for bypass
                spike_detected=False,   # No spike in bypass
                error_state=error_state
            )
            
        except Exception as e:
            logger.warning(f"Bypass path failed: {e}")
            # Fall back to normal processing
            return self._execute_cycle(question, verbose=False)
    
    def _search_memory(self, question: str) -> Dict[str, Any]:
        """Search episodic memory for relevant documents."""
        try:
            # Get configuration parameters
            max_docs = getattr(self.config, 'max_retrieved_docs', 10)
            enable_graph_search = getattr(self.config, 'enable_graph_search', False)
            
            # Get query embedding
            query_embedding = self.l2_memory._encode_text(question)
            
            if enable_graph_search and hasattr(self, 'current_graph') and self.current_graph:
                # Use graph-based search
                results = self.graph_memory_search.search_with_graph(
                    query_embedding=query_embedding,
                    graph=self.current_graph,
                    top_k=max_docs
                )
            else:
                # Use standard similarity search
                results = self.l2_memory.search_similar(query_embedding, top_k=max_docs)
            
            # Format results
            documents = []
            for result in results:
                documents.append({
                    "index": result.get("index", 0),
                    "text": result.get("text", ""),
                    "similarity": result.get("similarity", 0.0),
                    "c_value": result.get("c_value", 0.5),
                    "timestamp": result.get("timestamp", time.time())
                })
            
            return {
                "documents": documents,
                "total_found": len(documents),
                "search_type": "graph" if enable_graph_search else "similarity"
            }
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return {"documents": [], "total_found": 0, "search_type": "fallback"}
    
    def _analyze_graph(
        self, 
        question: str, 
        retrieved_docs: List[Dict[str, Any]], 
        error_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform graph analysis."""
        
        if not self.l3_graph:
            return {"available": False, "reason": "Graph reasoner not available"}
        
        try:
            return self.l3_graph.analyze_and_reason(
                question=question,
                episodes=retrieved_docs,
                previous_state=self.previous_state,
                uncertainty_state=error_state
            )
        except Exception as e:
            logger.warning(f"Graph analysis failed: {e}")
            # Use fallback registry
            return execute_fallback(
                FallbackReason.FULL_CALCULATION_FAILED,
                e,
                {
                    'question': question,
                    'documents': retrieved_docs,
                    'error_state': error_state
                }
            )
    
    def _detect_spike_via_pipeline(
        self, 
        graph_analysis: Dict[str, Any], 
        retrieved_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Detect spikes using the new pipeline."""
        
        try:
            # Extract geDIG result from graph analysis
            gedig_result = graph_analysis.get('gedig_result', {
                'gedig': 0.0, 'ged': 0.0, 'ig': 0.0, 'mode': 'fallback'
            })
            
            # Execute spike detection pipeline
            pipeline_result = self.spike_pipeline.execute(
                gedig_result=gedig_result,
                graph_analysis=graph_analysis,
                retrieved_docs=retrieved_docs,
                previous_state=self.previous_state,
                context={'insight_registry': self.insight_registry}
            )
            
            return pipeline_result.formatted_result
            
        except Exception as e:
            logger.error(f"Spike detection failed: {e}")
            return {
                'spike_detected': False,
                'error': str(e),
                'confidence': 0.0
            }
    
    def _generate_response(
        self,
        question: str,
        retrieved_docs: List[Dict[str, Any]],
        graph_analysis: Dict[str, Any],
        spike_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate LLM response."""
        
        try:
            # Prepare context
            context_docs = [doc.get("text", "") for doc in retrieved_docs[:5]]
            context_str = "\n".join(f"- {doc}" for doc in context_docs if doc)
            
            # Adjust prompt based on spike detection
            if spike_result.get("spike_detected", False):
                prompt = (f"INSIGHT DETECTED! Confidence: {spike_result.get('confidence', 0):.2f}\n"
                         f"Question: {question}\n"
                         f"Relevant context:\n{context_str}\n"
                         f"Provide an insightful response that connects these concepts:")
            else:
                prompt = (f"Question: {question}\n"
                         f"Context:\n{context_str}\n"
                         f"Please provide a helpful response:")
            
            return self.l4_llm.generate(
                prompt=prompt,
                max_tokens=getattr(self.config, 'max_response_tokens', 500)
            )
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                "text": f"I apologize, but I encountered an error while generating a response: {str(e)}",
                "error": str(e)
            }
    
    def _calculate_quality(
        self, 
        spike_result: Dict[str, Any], 
        memory_results: Dict[str, Any]
    ) -> float:
        """Calculate reasoning quality score."""
        
        base_quality = 0.5
        
        # Boost for spike detection
        if spike_result.get("spike_detected", False):
            base_quality += 0.3 * spike_result.get("confidence", 0)
        
        # Boost for good memory retrieval
        doc_count = memory_results.get("total_found", 0)
        if doc_count > 0:
            base_quality += min(0.2, doc_count * 0.05)
        
        # Penalty for errors
        if "error" in spike_result or "error" in memory_results:
            base_quality *= 0.7
        
        return min(max(base_quality, 0.0), 1.0)
    
    def _update_state(self, cycle_result: CycleResult):
        """Update agent state after cycle."""
        self.previous_state = {
            "last_question": cycle_result.question,
            "last_spike": cycle_result.spike_detected,
            "last_quality": cycle_result.reasoning_quality,
            "graph_state": cycle_result.graph_analysis,
            "cycle_count": self.cycle_count
        }
    
    def _save_query(self, question: str, result: CycleResult, processing_time: float):
        """Save query using query recorder if available."""
        
        if not self.query_recorder:
            return
        
        try:
            # Get query embedding
            query_vec = self.l1_embedder.get_embedding(question)
            
            # Build record
            record = self.query_recorder.build_record(
                text=question,
                response=result.response,
                has_spike=result.spike_detected,
                query_vec=query_vec,
                processing_time=processing_time,
                total_cycles=1,
                converged=True,
                reasoning_quality=result.reasoning_quality,
                retrieved_doc_count=len(result.retrieved_documents),
                llm_provider=self.l4_llm.__class__.__name__
            )
            
            self.query_recorder.save([record])
            
        except Exception as e:
            logger.warning(f"Query save failed: {e}")
    
    def _create_error_result(self, question: str, error: str) -> CycleResult:
        """Create error result when cycle fails."""
        return CycleResult(
            question=question,
            retrieved_documents=[],
            graph_analysis={},
            response=f"I apologize, but I encountered an error: {error}",
            reasoning_quality=0.0,
            spike_detected=False,
            error_state={"error": error}
        )
    
    # Public utility methods
    def add_knowledge(self, text: str, c_value: float = 0.5) -> Dict[str, Any]:
        """Add knowledge to the agent's memory."""
        try:
            episode_idx = self.l2_memory.store_episode(text, c_value)
            return {
                "success": episode_idx >= 0,
                "episode_index": episode_idx,
                "text_length": len(text)
            }
        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}")
            return {"success": False, "error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "cycles_executed": self.cycle_count,
            "initialized": self._initialized,
            "memory_episodes": len(getattr(self.l2_memory, 'episodes', [])),
            "reasoning_history_length": len(self.reasoning_history),
            "spike_pipeline_metrics": self.spike_pipeline.get_pipeline_metrics(),
            "components": {
                "l1_available": self.l1_error_monitor is not None,
                "l2_available": self.l2_memory is not None,
                "l3_available": self.l3_graph is not None,
                "l4_available": self.l4_llm is not None,
            }
        }
    
    def reset_state(self):
        """Reset agent state."""
        self.cycle_count = 0
        self.previous_state = {}
        self.reasoning_history = []
        self.spike_pipeline.reset_pipeline()
        logger.info("Agent state reset completed")


# Backward compatibility function
def create_slim_agent(config=None, datastore=None) -> SlimMainAgent:
    """Create a SlimMainAgent instance."""
    return SlimMainAgent(config, datastore)


__all__ = ['SlimMainAgent', 'CycleResult', 'create_slim_agent']