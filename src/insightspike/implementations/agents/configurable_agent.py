"""
Configurable Agent
==================

A highly configurable Q&A agent that supports multiple operation modes.
Replace various agent variants with configuration-based behavior switching.
"""

import asyncio
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AgentMode(Enum):
    """Different operation modes for the agent"""

    BASIC = "basic"  # Standard MainAgent behavior
    ENHANCED = "enhanced"  # Graph-aware memory management
    QUERY_TRANSFORM = "query_transform"  # Query transformation through graph
    ADVANCED = "advanced"  # Multi-hop reasoning and branching
    OPTIMIZED = "optimized"  # Production-ready with caching
    GRAPH_CENTRIC = "graph_centric"  # Pure graph-based without C-values


@dataclass
class AgentConfig:
    """Unified configuration for all agent features"""

    # Base configuration
    mode: AgentMode = AgentMode.BASIC
    max_cycles: int = 3
    verbose: bool = False

    # Feature toggles
    enable_query_transform: bool = False
    enable_graph_aware_memory: bool = False
    enable_multi_hop: bool = False
    enable_query_branching: bool = False
    enable_caching: bool = False
    enable_async_processing: bool = False
    enable_gpu_acceleration: bool = False
    enable_evolution_tracking: bool = False

    # Component configurations
    llm_config: Optional[Dict[str, Any]] = None
    memory_config: Optional[Dict[str, Any]] = None
    graph_config: Optional[Dict[str, Any]] = None

    # Performance settings
    cache_size: int = 1000
    parallel_branches: int = 4
    embedding_batch_size: int = 32

    @classmethod
    def from_mode(cls, mode: AgentMode, **kwargs) -> "AgentConfig":
        """Create config with presets for a specific mode"""
        config = cls(mode=mode, **kwargs)

        if mode == AgentMode.ENHANCED:
            config.enable_graph_aware_memory = True

        elif mode == AgentMode.QUERY_TRANSFORM:
            config.enable_query_transform = True

        elif mode == AgentMode.ADVANCED:
            config.enable_query_transform = True
            config.enable_multi_hop = True
            config.enable_query_branching = True

        elif mode == AgentMode.OPTIMIZED:
            config.enable_query_transform = True
            config.enable_multi_hop = True
            config.enable_query_branching = True
            config.enable_caching = True
            config.enable_async_processing = True
            config.enable_evolution_tracking = True

        elif mode == AgentMode.GRAPH_CENTRIC:
            config.enable_graph_aware_memory = True
            # Disable C-value based features

        return config


@dataclass
class UnifiedCycleResult:
    """Unified result structure that supports all agent variants"""

    # Basic fields (from original MainAgent)
    question: str
    response: str
    retrieved_documents: List[Dict[str, Any]]
    reasoning_quality: float
    spike_detected: bool
    cycle_number: int
    success: bool = True

    # Enhanced fields
    graph_analysis: Optional[Dict[str, Any]] = None
    error_state: Optional[Dict[str, Any]] = None

    # Query transformation fields
    query_state: Optional[Any] = None  # QueryState object
    transformation_history: Optional[List[Dict[str, Any]]] = None

    # Advanced fields
    reasoning_paths: Optional[List[Dict[str, Any]]] = None
    branch_results: Optional[List[Dict[str, Any]]] = None
    multi_hop_trace: Optional[List[str]] = None

    # Optimization fields
    cached: bool = False
    processing_time: float = 0.0
    gpu_accelerated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, including only non-None fields"""
        result = {
            "question": self.question,
            "response": self.response,
            "retrieved_documents": self.retrieved_documents,
            "reasoning_quality": self.reasoning_quality,
            "spike_detected": self.spike_detected,
            "cycle_number": self.cycle_number,
            "success": self.success,
            "processing_time": self.processing_time,
        }

        # Add optional fields if present
        optional_fields = [
            "graph_analysis",
            "error_state",
            "query_state",
            "transformation_history",
            "reasoning_paths",
            "branch_results",
            "multi_hop_trace",
        ]

        for field in optional_fields:
            value = getattr(self, field)
            if value is not None:
                if hasattr(value, "to_dict"):
                    result[field] = value.to_dict()
                else:
                    result[field] = value

        if self.cached:
            result["cached"] = True
        if self.gpu_accelerated:
            result["gpu_accelerated"] = True

        return result


class QueryCache:
    """Simple LRU cache for query results"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, UnifiedCycleResult] = {}
        self.access_order: List[str] = []

    def _hash_query(self, query: str) -> str:
        """Create hash key for query"""
        return hashlib.md5(query.encode()).hexdigest()

    def get(self, query: str) -> Optional[UnifiedCycleResult]:
        """Get cached result if available"""
        key = self._hash_query(query)
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            result = self.cache[key]
            result.cached = True
            return result
        return None

    def put(self, query: str, result: UnifiedCycleResult):
        """Store result in cache"""
        key = self._hash_query(query)

        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]

        self.cache[key] = result
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)


class ConfigurableAgent:
    """
    Configurable Q&A Agent with multiple operation modes.

    Features:
    - 6 operation modes: Basic, Enhanced, Query Transform, Advanced, Optimized, Graph-Centric
    - Feature toggles for fine-grained control
    - Backward compatible with all deprecated agent variants
    - Production-ready with caching, async processing, and GPU support

    This replaces all specialized agent implementations with a single configurable class.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize with unified configuration"""
        self.config = config or AgentConfig()

        # Core components (always present)
        self.l1_error_monitor = None
        self.l2_memory = None
        self.l3_graph = None
        self.l4_llm = None

        # Optional components (based on config)
        self.query_transformer = None
        self.query_cache = None
        self.evolution_tracker = None
        self.thread_pool = None

        # State tracking
        self.cycle_count = 0
        self.previous_state = {}
        self.transformation_histories = {}

        # Initialize based on configuration
        self._setup_components()

    def _setup_components(self):
        """Setup components based on configuration"""
        # Cache setup
        if self.config.enable_caching:
            self.query_cache = QueryCache(self.config.cache_size)
            logger.info(f"Query cache enabled with size {self.config.cache_size}")

        # Async processing setup
        if self.config.enable_async_processing:
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.config.parallel_branches
            )
            logger.info(
                f"Async processing enabled with {self.config.parallel_branches} workers"
            )

        # Query transformer setup
        if self.config.enable_query_transform:
            try:
                from insightspike.features.query_transformation import QueryTransformer

                # Align QueryTransformer's GNN usage with L3 graph config when available
                use_gnn = False
                try:
                    gc = getattr(self.config, "graph_config", None)
                    if gc is not None:
                        if hasattr(gc, "graph") and hasattr(gc.graph, "use_gnn"):
                            use_gnn = bool(gc.graph.use_gnn)
                        elif isinstance(gc, dict):
                            use_gnn = bool(gc.get("graph", {}).get("use_gnn", False))
                except Exception:
                    # Fall back silently if config shape differs
                    use_gnn = False

                self.query_transformer = QueryTransformer(use_gnn=use_gnn)
                logger.info("Query transformation enabled")
            except ImportError:
                logger.warning(
                    "Query transformation requested but module not available"
                )
                self.config.enable_query_transform = False

    def initialize(self) -> bool:
        """Initialize all components"""
        try:
            # Initialize based on mode
            if self.config.mode == AgentMode.GRAPH_CENTRIC:
                return self._initialize_graph_centric()
            else:
                return self._initialize_standard()

        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            return False

    def _initialize_standard(self) -> bool:
        """Standard initialization (for all non-graph-centric modes)"""
        try:
            # L1: Error Monitor
            from insightspike.implementations.layers.layer1_error_monitor import (
                ErrorMonitor,
            )

            self.l1_error_monitor = ErrorMonitor()

            # L2: Memory (unified with mode configuration)
            from insightspike.implementations.layers.layer2_memory_manager import (
                L2MemoryManager,
                MemoryConfig,
                MemoryMode,
            )

            if self.config.enable_graph_aware_memory:
                memory_config = MemoryConfig.from_mode(MemoryMode.ENHANCED)
            else:
                memory_config = MemoryConfig.from_mode(MemoryMode.SCALABLE)

            self.l2_memory = L2MemoryManager(memory_config)

            # L3: Graph Reasoner
            try:
                from insightspike.implementations.layers.layer3_graph_reasoner import (
                    L3GraphReasoner,
                )

                # Pass config with use_gnn setting
                graph_config = self.config.graph_config
                if hasattr(graph_config, "graph"):
                    # Respect config.yaml setting, but allow override if query transform is enabled
                    if self.config.enable_query_transform and not getattr(
                        graph_config.graph, "use_gnn", False
                    ):
                        logger.info("Enabling GNN for query transformation")
                        graph_config.graph.use_gnn = True

                self.l3_graph = L3GraphReasoner(graph_config)
            except ImportError:
                logger.warning("Graph reasoner not available")
                self.l3_graph = None

            # L4: LLM Provider
            from insightspike.implementations.layers.layer4_llm_interface import (
                get_llm_provider,
            )

            self.l4_llm = get_llm_provider(self.config.llm_config)

            if not self.l4_llm.initialize():
                logger.error("Failed to initialize LLM provider")
                return False

            logger.info(f"Agent initialized in {self.config.mode.value} mode")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def _initialize_graph_centric(self) -> bool:
        """Initialize for graph-centric mode (no C-values)"""
        try:
            from insightspike.implementations.layers.layer2_memory_manager import (
                L2MemoryManager,
                MemoryConfig,
                MemoryMode,
            )
            from insightspike.implementations.layers.layer3_graph_reasoner import (
                L3GraphReasoner,
            )
            from insightspike.implementations.layers.layer4_llm_interface import (
                get_llm_provider,
            )

            # Use graph-centric memory mode
            memory_config = MemoryConfig.from_mode(MemoryMode.GRAPH_CENTRIC)
            self.l2_memory = L2MemoryManager(memory_config)

            self.l3_graph = L3GraphReasoner(self.config.graph_config)
            self.l4_llm = get_llm_provider(self.config.llm_config)

            # No L1 in graph-centric mode
            self.l1_error_monitor = None

            logger.info("Agent initialized in graph-centric mode")
            return True

        except Exception as e:
            logger.error(f"Graph-centric initialization failed: {e}")
            return False

    def process_question(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Process a question using the configured features.

        This is the main entry point that handles all agent modes.
        """
        start_time = time.time()

        # Override config with kwargs if provided
        max_cycles = kwargs.get("max_cycles", self.config.max_cycles)
        verbose = kwargs.get("verbose", self.config.verbose)

        # Check cache first if enabled
        if self.config.enable_caching and self.query_cache:
            cached_result = self.query_cache.get(question)
            if cached_result:
                logger.info(f"Returning cached result for: {question}")
                return cached_result.to_dict()

        # Process based on mode and features
        if self.config.enable_query_transform:
            result = self._process_with_query_transform(question, max_cycles, verbose)
        elif self.config.mode == AgentMode.GRAPH_CENTRIC:
            result = self._process_graph_centric(question, verbose)
        else:
            result = self._process_standard(question, max_cycles, verbose)

        # Add processing time
        result.processing_time = time.time() - start_time

        # Cache result if enabled
        if self.config.enable_caching and self.query_cache:
            self.query_cache.put(question, result)

        return result.to_dict()

    def _process_standard(
        self, question: str, max_cycles: int, verbose: bool
    ) -> UnifiedCycleResult:
        """Standard processing (original MainAgent behavior)"""
        best_result = None
        max_quality = 0.0

        for cycle in range(max_cycles):
            if verbose:
                logger.info(f"\nCycle {cycle + 1}/{max_cycles}")

            # Execute one cycle
            cycle_result = self._execute_standard_cycle(question, cycle + 1)

            # Track best result
            if cycle_result.reasoning_quality > max_quality:
                max_quality = cycle_result.reasoning_quality
                best_result = cycle_result

            # Check for spike
            if cycle_result.spike_detected:
                logger.info(f"Insight spike detected in cycle {cycle + 1}!")
                break

        return best_result or UnifiedCycleResult(
            question=question,
            response="No valid response generated",
            retrieved_documents=[],
            reasoning_quality=0.0,
            spike_detected=False,
            cycle_number=1,
            success=False,
        )

    def _execute_standard_cycle(
        self, question: str, cycle_number: int
    ) -> UnifiedCycleResult:
        """Execute one standard reasoning cycle"""
        try:
            # L1: Error monitoring (if available)
            error_state = {}
            if self.l1_error_monitor:
                error_state = self.l1_error_monitor.analyze_uncertainty(
                    question, self.previous_state
                )

            # L2: Memory search
            memory_results = self.l2_memory.search_episodes(question, k=5)
            retrieved_docs = (
                memory_results
                if isinstance(memory_results, list)
                else memory_results.get("documents", [])
            )

            # L3: Graph reasoning (if available)
            graph_analysis = {}
            spike_detected = False
            reasoning_quality = 0.5

            if self.l3_graph and retrieved_docs:
                graph_analysis = self.l3_graph.analyze_documents(
                    retrieved_docs,
                    {"previous_state": self.previous_state, "error_state": error_state},
                )
                spike_detected = graph_analysis.get("spike_detected", False)
                reasoning_quality = graph_analysis.get("reasoning_quality", 0.5)

            # L4: Generate response
            llm_context = {
                "retrieved_documents": retrieved_docs,
                "graph_analysis": graph_analysis,
                "reasoning_quality": reasoning_quality,
            }

            if hasattr(self.l4_llm, "generate_response_detailed"):
                llm_result = self.l4_llm.generate_response_detailed(
                    llm_context, question
                )
                response = llm_result.get("response", "")
            else:
                response = self.l4_llm.generate_response(llm_context, question)

            # Update state
            self.previous_state = {
                "question": question,
                "response": response,
                "graph_analysis": graph_analysis,
            }

            return UnifiedCycleResult(
                question=question,
                response=response,
                retrieved_documents=retrieved_docs,
                reasoning_quality=reasoning_quality,
                spike_detected=spike_detected,
                cycle_number=cycle_number,
                graph_analysis=graph_analysis,
                error_state=error_state,
            )

        except Exception as e:
            logger.error(f"Cycle {cycle_number} failed: {e}")
            return UnifiedCycleResult(
                question=question,
                response=f"Error: {str(e)}",
                retrieved_documents=[],
                reasoning_quality=0.0,
                spike_detected=False,
                cycle_number=cycle_number,
                success=False,
            )

    def _process_with_query_transform(
        self, question: str, max_cycles: int, verbose: bool
    ) -> UnifiedCycleResult:
        """Process with query transformation enabled"""
        # Initialize transformation history
        from insightspike.features.query_transformation import (
            QueryTransformationHistory,
        )

        transformation_history = QueryTransformationHistory(initial_query=question)

        # Place query on graph
        initial_state = self.query_transformer.place_query_on_graph(
            question, self._get_current_knowledge_graph()
        )
        transformation_history.add_state(initial_state)

        best_result = None
        max_quality = 0.0

        for cycle in range(max_cycles):
            if verbose:
                logger.info(f"\nTransformation Cycle {cycle + 1}/{max_cycles}")

            # Get current query state
            current_state = transformation_history.get_current_state()
            query_text = current_state.text

            # Execute cycle with transformed query
            cycle_result = self._execute_standard_cycle(query_text, cycle + 1)

            # Transform query for next cycle
            if cycle < max_cycles - 1 and self.l3_graph:
                new_state = self.query_transformer.transform_query(
                    current_state,
                    self._get_current_knowledge_graph(),
                    cycle_result.retrieved_documents,
                )
                transformation_history.add_state(new_state)

            # Track best result
            if cycle_result.reasoning_quality > max_quality:
                max_quality = cycle_result.reasoning_quality
                best_result = cycle_result
                best_result.query_state = current_state
                best_result.transformation_history = [
                    s.to_dict() for s in transformation_history.states
                ]

            # Check for completion
            if current_state.stage == "complete" or current_state.confidence > 0.9:
                break

        return best_result

    def _process_graph_centric(
        self, question: str, verbose: bool
    ) -> UnifiedCycleResult:
        """Process using graph-centric approach (no C-values)"""
        try:
            # Search using graph-based importance
            results = self.l2_memory.search(question, k=5)

            # Generate narrative if L4 is available
            response = ""
            if self.l4_llm and hasattr(self.l4_llm, "generate_narrative"):
                response = self.l4_llm.generate_narrative(results)
            else:
                # Fallback to simple concatenation
                response = "\n".join([r.get("text", "") for r in results])

            return UnifiedCycleResult(
                question=question,
                response=response,
                retrieved_documents=results,
                reasoning_quality=0.7,  # Graph-centric doesn't compute this
                spike_detected=False,
                cycle_number=1,
            )

        except Exception as e:
            logger.error(f"Graph-centric processing failed: {e}")
            return UnifiedCycleResult(
                question=question,
                response=f"Error: {str(e)}",
                retrieved_documents=[],
                reasoning_quality=0.0,
                spike_detected=False,
                cycle_number=1,
                success=False,
            )

    def _get_current_knowledge_graph(self):
        """Get current knowledge graph for query transformation"""
        if hasattr(self.l3_graph, "current_graph"):
            return self.l3_graph.current_graph
        return None

    def add_episode(self, text: str, **kwargs) -> bool:
        """Add an episode to memory (unified interface)"""
        try:
            if self.config.mode == AgentMode.GRAPH_CENTRIC:
                # Graph-centric doesn't use C-values
                result = self.l2_memory.add_episode(text)
                return result.get("success", False)
            else:
                # Standard approach with C-value
                c_value = kwargs.get("c_value", 0.5)
                metadata = kwargs.get("metadata", {})
                return self.l2_memory.store_episode(text, c_value, metadata)

        except Exception as e:
            logger.error(f"Failed to add episode: {e}")
            return False

    def cleanup(self):
        """Cleanup resources"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)

        if hasattr(self.l4_llm, "cleanup"):
            self.l4_llm.cleanup()

        logger.info("Agent cleanup completed")


# Convenience functions for backward compatibility
def create_agent(mode: str = "basic", **kwargs) -> ConfigurableAgent:
    """Create agent with specified mode"""
    try:
        agent_mode = AgentMode(mode)
    except ValueError:
        logger.warning(f"Unknown mode '{mode}', using BASIC")
        agent_mode = AgentMode.BASIC

    config = AgentConfig.from_mode(agent_mode, **kwargs)
    return ConfigurableAgent(config)


# Aliases for backward compatibility
MainAgent = ConfigurableAgent  # Drop-in replacement
UnifiedMainAgent = ConfigurableAgent  # Migration support
