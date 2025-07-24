"""
Main Agent Orchestrator
======================

Simplified main agent that coordinates all layers for clean execution.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ...core.base.datastore import DataStore
from ...core.episode import Episode

logger = logging.getLogger(__name__)

from ...config.models import InsightSpikeConfig  # Import Pydantic config
from ..layers.layer1_error_monitor import ErrorMonitor
from ..layers.layer2_compatibility import (
    CompatibleL2MemoryManager as Memory,  # Layer 2: Memory Manager with compatibility
)
from ..layers.layer4_llm_interface import get_llm_provider

try:
    from ..layers.layer3_graph_reasoner import L3GraphReasoner

    GRAPH_REASONER_AVAILABLE = True
except ImportError:
    GRAPH_REASONER_AVAILABLE = False
    logger.warning("Graph reasoner (Layer 3) not available - requires PyTorch")


@dataclass
class CycleResult:
    """Result from one reasoning cycle"""

    question: str
    retrieved_documents: List[Dict[str, Any]]
    graph_analysis: Dict[str, Any]
    response: str
    reasoning_quality: float
    spike_detected: bool
    error_state: Dict[str, Any]
    cycle_number: int
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility"""
        return {
            "question": self.question,
            "documents": self.retrieved_documents,
            "graph": self.graph_analysis.get("graph"),
            "response": self.response,
            "metrics": self.graph_analysis.get("metrics", {}),
            "conflicts": self.graph_analysis.get("conflicts", {}),
            "reward": self.graph_analysis.get("reward", {}),
            "spike_detected": self.spike_detected,
            "reasoning_quality": self.reasoning_quality,
            "cycle": self.cycle_number,
            "success": self.success,
        }


class MainAgent:
    """
    Main orchestrating agent that coordinates all layers.

    Simplified architecture:
    - L1: Error monitoring and uncertainty
    - L2: Memory search and update
    - L3: Graph reasoning and spike detection
    - L4: Language generation
    """

    def __init__(self, config=None, datastore: Optional[DataStore] = None):
        """
        Initialize MainAgent with injected dependencies.

        Args:
            config: Configuration object (InsightSpikeConfig or legacy format)
            datastore: DataStore instance for persistence
        """
        # Store config - should be provided by DependencyFactory
        if config is None:
            raise ValueError("Config must be provided to MainAgent")

        # Handle both Pydantic and legacy config
        if isinstance(config, InsightSpikeConfig):
            self.config = config
            self.is_pydantic_config = True
        else:
            # Legacy config support
            self.config = config
            self.is_pydantic_config = False

        # Store the datastore for persistence operations
        self.datastore = datastore

        # Initialize layers
        self.l1_error_monitor = ErrorMonitor(self.config)

        # Initialize Layer 2 Memory Manager with backward compatibility
        if self.is_pydantic_config:
            memory_dim = self.config.embedding.dimension
        else:
            memory_dim = 384
            if hasattr(self.config, "embedding") and hasattr(
                self.config.embedding, "dimension"
            ):
                memory_dim = self.config.embedding.dimension
        self.l2_memory = Memory(dim=memory_dim, config=self.config)

        self.l3_graph = (
            L3GraphReasoner(self.config) if GRAPH_REASONER_AVAILABLE else None
        )
        self.l4_llm = get_llm_provider(self.config, safe_mode=False)

        # State tracking
        self.cycle_count = 0
        self.previous_state = {}
        self.reasoning_history = []

        self._initialized = False

    def initialize(self) -> bool:
        """Initialize all components"""
        try:
            logger.info("Initializing MainAgent components...")

            # Check if LLM is already initialized (from cache)
            if hasattr(self.l4_llm, "initialized") and self.l4_llm.initialized:
                logger.info("LLM provider already initialized (cached)")
            else:
                # Initialize LLM with safe fallback
                try:
                    if not self.l4_llm.initialize():
                        logger.error("Failed to initialize LLM provider")
                        return False
                except Exception as e:
                    logger.error(f"LLM initialization failed: {e}")
                    # Re-raise the error to see what's actually happening
                    raise

            # Try to load existing memory
            if not self.l2_memory.load():
                logger.info("No existing memory found, starting fresh")

            # Initialize error monitor
            self.l1_error_monitor.reset()

            self._initialized = True
            logger.info("MainAgent initialization complete")
            return True

        except Exception as e:
            logger.error(f"MainAgent initialization failed: {e}")
            return False

    def process_question(
        self, question: str, max_cycles: int = 10, verbose: bool = False
    ) -> CycleResult:
        """
        Process a question through multiple reasoning cycles until convergence.

        Args:
            question: The question to process
            max_cycles: Maximum number of reasoning cycles
            verbose: Whether to log detailed information

        Returns:
            CycleResult containing complete results and reasoning trace
        """
        if not self._initialized:
            if not self.initialize():
                return self._error_cycle_result(
                    question, 0, "Failed to initialize agent"
                )

        results = []
        convergence_reached = False

        for cycle in range(max_cycles):
            try:
                # Execute one reasoning cycle
                cycle_result = self._execute_cycle(question, verbose=verbose)
                results.append(cycle_result)

                if verbose:
                    logger.info(
                        f"Cycle {cycle + 1}: Quality={cycle_result.reasoning_quality:.3f}, "
                        f"Spike={cycle_result.spike_detected}"
                    )

                # Check for convergence or high-quality answer
                if (
                    cycle_result.reasoning_quality > 0.8
                    or cycle_result.spike_detected
                    or cycle > 0
                    and self._check_convergence(results[-2:], question)
                ):
                    convergence_reached = True
                    break

            except Exception as e:
                logger.error(f"Cycle {cycle + 1} failed: {e}")
                cycle_result = self._error_cycle_result(question, cycle + 1, str(e))
                results.append(cycle_result)
                break

        # Compile results
        result = self._compile_results(results, convergence_reached, verbose)

        # Update reasoning history
        self.reasoning_history.append(
            {
                "question": question,
                "cycles": len(results),
                "quality": result.reasoning_quality,
                "converged": convergence_reached,
            }
        )

        return result

    def _execute_cycle(self, question: str, verbose: bool = False) -> CycleResult:
        """Execute one complete reasoning cycle"""
        self.cycle_count += 1

        try:
            # L1: Error monitoring and uncertainty calculation
            error_state = self.l1_error_monitor.analyze_uncertainty(
                question, self.previous_state
            )

            # L2: Memory search and retrieval
            memory_results = self._search_memory(question)
            retrieved_docs = memory_results["documents"]

            logger.debug(f"Memory search returned {len(retrieved_docs)} documents")

            # L3: Graph reasoning and analysis
            graph_context = {
                "previous_state": self.previous_state,
                "error_state": error_state,
                "memory_stats": memory_results.get("stats", {}),
                "previous_graph": self.previous_state.get("graph_state")
                if self.previous_state
                else None,
            }

            # L3: Graph analysis (optional)
            if self.l3_graph:
                logger.debug(
                    f"L3GraphReasoner available, processing {len(retrieved_docs)} documents"
                )
                graph_analysis = self.l3_graph.analyze_documents(
                    retrieved_docs, graph_context
                )
                logger.debug(
                    f"Graph analysis result: {graph_analysis.keys() if graph_analysis else 'None'}"
                )
            else:
                logger.warning("L3GraphReasoner not available")
                # Fallback when graph reasoner is not available
                graph_analysis = {
                    "graph": None,
                    "metrics": {"delta_ged": 0.0, "delta_ig": 0.0},
                    "conflicts": {"total_conflicts": 0, "conflict_types": {}},
                    "reward": {"insight_reward": 0.0, "quality_bonus": 0.0},
                    "reasoning_quality": 0.5,  # Neutral quality without graph analysis
                    "spike_detected": False,
                }

            # L4: Language generation
            llm_context = {
                "retrieved_documents": retrieved_docs,
                "graph_analysis": graph_analysis,
                "previous_state": self.previous_state,
                "reasoning_quality": graph_analysis.get("reasoning_quality", 0.0),
            }

            # Call generate_response_detailed to get full result dict
            if hasattr(self.l4_llm, "generate_response_detailed"):
                llm_result = self.l4_llm.generate_response_detailed(
                    llm_context, question
                )
            else:
                # Fallback for simple providers - wrap string result
                response = self.l4_llm.generate_response(llm_context, question)
                llm_result = {"response": response, "success": True, "confidence": 0.5}

            # Calculate overall reasoning quality
            reasoning_quality = self._calculate_reasoning_quality(
                error_state, memory_results, graph_analysis, llm_result
            )

            # Create cycle result
            cycle_result = CycleResult(
                question=question,
                retrieved_documents=retrieved_docs,
                graph_analysis=graph_analysis,
                response=llm_result.get("response", ""),
                reasoning_quality=reasoning_quality,
                spike_detected=graph_analysis.get("spike_detected", False),
                error_state=error_state,
                cycle_number=self.cycle_count,
                success=llm_result.get("success", True),
            )

            # Update memory with reward signal if spike detected
            if cycle_result.spike_detected:
                self._update_memory_rewards(retrieved_docs, graph_analysis)

            # Store question and response in memory for future retrieval
            memory_text = f"Q: {question}\nA: {cycle_result.response}"

            # Use L2MemoryManager's store_episode method which properly uses SentenceTransformer
            try:
                success = self.l2_memory.store_episode(
                    memory_text, c_value=reasoning_quality
                )
                if success:
                    logger.debug(
                        f"Stored episode in memory: {len(memory_text)} chars with quality {reasoning_quality:.3f}"
                    )
                else:
                    logger.warning("Failed to store episode in memory")

            except Exception as e:
                logger.warning(f"Failed to add to memory: {e}")

            # Update previous state for next cycle
            self.previous_state = {
                "last_response": cycle_result.response,
                "reasoning_quality": reasoning_quality,
                "graph_state": graph_analysis.get("graph"),
                "cycle_count": self.cycle_count,
            }

            return cycle_result

        except Exception as e:
            logger.error(f"Cycle execution failed: {e}")
            return self._error_cycle_result(question, self.cycle_count, str(e))

    def _search_memory(self, question: str) -> Dict[str, Any]:
        """Search episodic memory for relevant documents"""
        try:
            # Use L2MemoryManager's search_episodes method which properly uses SentenceTransformer
            if self.is_pydantic_config:
                max_docs = self.config.memory.max_retrieved_docs
            else:
                max_docs = getattr(self.config.memory, "max_retrieved_docs", 10)

            results = self.l2_memory.search_episodes(
                question,
                k=max_docs,
            )

            # Convert search_episodes results to documents format
            documents = []
            for result in results:
                # Get the actual episode to include embedding
                episode_idx = result["index"]
                if 0 <= episode_idx < len(self.l2_memory.episodes):
                    episode = self.l2_memory.episodes[episode_idx]
                    documents.append(
                        {
                            "text": result["text"],
                            "similarity": result["similarity"],
                            "index": result["index"],
                            "c_value": result["c_value"],
                            "timestamp": result.get("timestamp", time.time()),
                            "embedding": episode.vec,  # Include the embedding vector
                        }
                    )
                else:
                    # Fallback without embedding
                    documents.append(
                        {
                            "text": result["text"],
                            "similarity": result["similarity"],
                            "index": result["index"],
                            "c_value": result["c_value"],
                            "timestamp": result.get("timestamp", time.time()),
                        }
                    )

            stats = self.l2_memory.get_memory_stats()

            return {"documents": documents, "stats": stats, "success": True}

        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return {"documents": [], "stats": {}, "success": False, "error": str(e)}

    def _calculate_reasoning_quality(
        self,
        error_state: Dict,
        memory_results: Dict,
        graph_analysis: Dict,
        llm_result: Dict,
    ) -> float:
        """Calculate overall reasoning quality score"""
        try:
            # Error component (lower error = higher quality)
            error_score = 1.0 - error_state.get("uncertainty", 0.5)

            # Memory component (more relevant docs = higher quality)
            memory_score = min(1.0, len(memory_results.get("documents", [])) / 3)

            # Graph component
            graph_score = graph_analysis.get("reasoning_quality", 0.0)

            # LLM component
            llm_score = llm_result.get("confidence", 0.5)

            # Weighted combination
            weights = [0.2, 0.3, 0.3, 0.2]  # error, memory, graph, llm
            scores = [error_score, memory_score, graph_score, llm_score]

            quality = sum(w * s for w, s in zip(weights, scores))
            return max(0.0, min(1.0, quality))

        except Exception as e:
            logger.error(f"Quality calculation failed: {e}")
            return 0.3  # Default moderate quality

    def _update_memory_rewards(self, documents: List[Dict], graph_analysis: Dict):
        """Update memory C-values based on reasoning rewards"""
        try:
            reward_info = graph_analysis.get("reward", {})
            total_reward = reward_info.get("total", 0.0)

            if total_reward > 0:
                # Increase C-values for retrieved documents
                episode_indices = []
                for doc in documents:
                    if "index" in doc:
                        current_c = doc.get("c_value", 0.5)
                        boost = min(0.1, total_reward * 0.05)  # Small positive boost
                        new_c = current_c + boost

                        self.l2_memory.update_c_value(doc["index"], new_c)
                        episode_indices.append(doc["index"])

                # Check for automatic episode management based on graph metrics
                self._check_episode_management_triggers(graph_analysis, episode_indices)

                logger.debug(f"Updated memory rewards with boost: {total_reward:.3f}")

        except Exception as e:
            logger.error(f"Memory reward update failed: {e}")

    def _check_episode_management_triggers(
        self, graph_analysis: Dict, episode_indices: List[int]
    ):
        """Check if graph metrics trigger episode merge/split/prune operations"""
        try:
            metrics = graph_analysis.get("metrics", {})
            conflicts = graph_analysis.get("conflicts", {})

            # Get threshold values from config
            merge_threshold = getattr(self.config.graph, "episode_merge_threshold", 0.8)
            split_threshold = getattr(self.config.graph, "episode_split_threshold", 0.3)
            prune_threshold = getattr(self.config.graph, "episode_prune_threshold", 0.1)

            # High similarity + low conflict might trigger merge
            delta_ged = metrics.get("delta_ged", 0.0)
            total_conflicts = conflicts.get("total", 0.0)

            if delta_ged < 0.2 and total_conflicts < 0.3 and len(episode_indices) >= 2:
                # Similarity is high, conflict is low - consider merging
                if hasattr(self.l2_memory, "get_episode_similarity"):
                    similarities = self.l2_memory.get_episode_similarity(
                        episode_indices
                    )
                    if max(similarities) > merge_threshold:
                        logger.info(
                            f"Graph analysis suggests merging episodes {episode_indices[:2]}"
                        )
                        merged_idx = self.l2_memory.merge_episodes(episode_indices[:2])
                        if merged_idx >= 0:
                            logger.info(f"Auto-merged episodes to index {merged_idx}")

            # High conflict or low quality might trigger split
            elif total_conflicts > 0.7 or delta_ged > split_threshold:
                for idx in episode_indices:
                    episode = (
                        self.l2_memory.episodes[idx]
                        if idx < len(self.l2_memory.episodes)
                        else None
                    )
                    if (
                        episode and len(episode.text.split(".")) > 2
                    ):  # Has multiple sentences
                        logger.info(f"Graph analysis suggests splitting episode {idx}")
                        split_indices = self.l2_memory.split_episode(idx)
                        if split_indices:
                            logger.info(
                                f"Auto-split episode {idx} into {split_indices}"
                            )
                        break

            # Very low C-values might trigger pruning
            memory_stats = self.l2_memory.get_memory_stats()
            if memory_stats.get("c_value_min", 1.0) < prune_threshold:
                logger.info("Graph analysis suggests pruning low-value episodes")
                pruned_count = self.l2_memory.prune_low_value_episodes(
                    prune_threshold * 2
                )  # Conservative pruning
                if pruned_count > 0:
                    logger.info(f"Auto-pruned {pruned_count} low-value episodes")

        except Exception as e:
            logger.error(f"Episode management trigger check failed: {e}")

    def _check_convergence(
        self, recent_results: List[CycleResult], question: str
    ) -> bool:
        """Check if reasoning has converged using semantic similarity"""
        if len(recent_results) < 2:
            return False

        try:
            # Check quality stability
            qualities = [r.reasoning_quality for r in recent_results]
            quality_diff = abs(qualities[-1] - qualities[-2])

            # Check response similarity using semantic embeddings
            responses = [r.response for r in recent_results]

            # Create embedder for semantic comparison
            from ...processing.embedder import get_model

            embedder = get_model()
            embeddings = embedder.encode(responses)

            # Calculate cosine similarity between the last two responses
            from sklearn.metrics.pairwise import cosine_similarity

            similarity_matrix = cosine_similarity(embeddings)
            response_similarity = similarity_matrix[
                0, 1
            ]  # Similarity between first and second response

            # Convergence if quality is stable and responses are semantically similar
            converged = (
                quality_diff < 0.1 and response_similarity > 0.95
            )  # Higher threshold for semantic similarity

            if converged:
                logger.info("Reasoning convergence detected")

            return converged

        except Exception as e:
            logger.error(f"Convergence check failed: {e}")
            return False

    def _compile_results(
        self, results: List[CycleResult], converged: bool, verbose: bool
    ) -> CycleResult:
        """Compile results from all cycles"""
        if not results:
            return self._error_cycle_result("", 0, "No results generated")

        # Use the best result (highest quality)
        best_result = max(results, key=lambda r: r.reasoning_quality)

        # Create a new CycleResult with additional metadata
        # Store extra data in graph_analysis for backward compatibility
        enhanced_graph_analysis = best_result.graph_analysis.copy()
        enhanced_graph_analysis.update(
            {
                "total_cycles": len(results),
                "converged": converged,
                "cycle_history": [r.to_dict() for r in results] if verbose else [],
                "agent_stats": {
                    "memory_episodes": self.l2_memory.get_memory_stats().get(
                        "total_episodes", 0
                    ),
                    "total_processed": len(self.reasoning_history),
                },
            }
        )

        return CycleResult(
            question=best_result.question,
            retrieved_documents=best_result.retrieved_documents,
            graph_analysis=enhanced_graph_analysis,
            response=best_result.response,
            reasoning_quality=best_result.reasoning_quality,
            spike_detected=best_result.spike_detected,
            error_state=best_result.error_state,
            cycle_number=best_result.cycle_number,
            success=best_result.success,
        )

    def _error_result(self, question: str, error: str) -> Dict[str, Any]:
        """Generate error result"""
        return {
            "question": question,
            "response": f"I apologize, but I encountered an error: {error}",
            "success": False,
            "error": error,
            "reasoning_quality": 0.0,
            "spike_detected": False,
            "total_cycles": 0,
        }

    def _error_cycle_result(self, question: str, cycle: int, error: str) -> CycleResult:
        """Generate error cycle result"""
        return CycleResult(
            question=question,
            retrieved_documents=[],
            graph_analysis={},
            response=f"Error in cycle {cycle}: {error}",
            reasoning_quality=0.0,
            spike_detected=False,
            error_state={"error": error},
            cycle_number=cycle,
            success=False,
        )

    def add_document(
        self, text: str, c_value: float = 0.5, metadata: Optional[Dict] = None
    ) -> bool:
        """Add a document to memory"""
        episode_idx = self.l2_memory.store_episode(text, c_value, metadata)
        return episode_idx >= 0

    def add_knowledge(self, text: str, c_value: float = 0.5) -> Dict[str, Any]:
        """
        Add knowledge to the agent's memory.

        Args:
            text: The knowledge text to add
            c_value: Confidence value for the knowledge

        Returns:
            Dict containing success status and any error messages
        """
        try:
            # Use L2MemoryManager's store_episode (which already handles graph updates)
            episode_idx = self.l2_memory.store_episode(text, c_value)
            
            # Handle different return types from store_episode
            if episode_idx is None or (isinstance(episode_idx, int) and episode_idx < 0):
                # Try alternative approach - create episode directly
                logger.warning("store_episode failed, creating episode manually")
                from insightspike.core.episode import Episode
                
                # Get embedding
                embedding = None
                if hasattr(self.l2_memory, 'embedding_model') and self.l2_memory.embedding_model:
                    embeddings = self.l2_memory.embedding_model.encode(text)
                    # encode returns 2D array, get first element
                    if isinstance(embeddings, np.ndarray) and len(embeddings.shape) == 2:
                        embedding = embeddings[0]
                    else:
                        embedding = embeddings
                elif hasattr(self.l2_memory, '_get_embedding'):
                    embedding = self.l2_memory._get_embedding(text)
                
                if embedding is not None:
                    episode = Episode(
                        text=text,
                        vec=embedding,
                        c=c_value,
                        timestamp=time.time(),
                        metadata={"c_value": c_value}
                    )
                    self.l2_memory.episodes.append(episode)
                    episode_idx = len(self.l2_memory.episodes) - 1
                else:
                    raise Exception("Failed to create embedding for episode")

            # Get the last added episode
            if episode_idx is None:
                episode_idx = len(self.l2_memory.episodes) - 1
            episode = self.l2_memory.episodes[episode_idx]
            vector = episode.vec

            # Get graph state from Layer2's ScalableGraphManager
            graph_nodes = 0
            graph_analysis = None

            # Check if Layer2 is using ScalableGraphManager
            if (
                hasattr(self.l2_memory, "scalable_graph")
                and self.l2_memory.scalable_graph
            ):
                # Get current graph from Layer2
                current_graph = self.l2_memory.scalable_graph.get_current_graph()
                graph_nodes = current_graph.num_nodes if current_graph else 0

                # Only use Layer3 for analysis, not rebuilding
                if self.l3_graph and current_graph and graph_nodes > 0:
                    # Analyze the existing graph (no rebuilding)
                    graph_analysis = self.l3_graph.analyze_documents(
                        [],  # Empty documents - just analyze existing graph
                        context={"graph": current_graph},
                    )
                    logger.debug(f"Graph analysis on {graph_nodes} nodes")
            else:
                # Fallback to old behavior if not using ScalableGraphManager
                logger.warning(
                    "Layer2 not using ScalableGraphManager, falling back to full rebuild"
                )
                if self.l3_graph:
                    # This is the inefficient path we want to avoid
                    all_documents = []
                    for i, ep in enumerate(self.l2_memory.episodes):
                        all_documents.append(
                            {
                                "text": ep.text,
                                "embedding": ep.vec,
                                "c_value": getattr(ep, "c_value", 0.5),
                                "episode_idx": i,
                                "timestamp": getattr(ep, "timestamp", time.time()),
                            }
                        )
                    graph_analysis = self.l3_graph.analyze_documents(all_documents)
                    graph_nodes = (
                        graph_analysis["metrics"].get("graph_size_current", 0)
                        if graph_analysis and "metrics" in graph_analysis
                        else 0
                    )

            result = {
                "episode_idx": episode_idx,
                "vector": vector,
                "text": text,
                "c_value": c_value,
                "graph_analysis": graph_analysis,
                "total_episodes": len(self.l2_memory.episodes),
                "graph_nodes": graph_nodes,
                "success": True,
            }

            return result

        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}")
            return {"episode_idx": -1, "success": False, "error": str(e)}

    def add_episode_with_graph_update(
        self, text: str, c_value: float = 0.5
    ) -> Dict[str, Any]:
        """Deprecated: Use add_knowledge() instead."""
        import warnings

        warnings.warn(
            "add_episode_with_graph_update is deprecated. Use add_knowledge() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.add_knowledge(text, c_value)

    def get_memory_graph_state(self) -> Dict[str, Any]:
        """Get current state of memory and graph for analysis."""
        try:
            memory_stats = self.l2_memory.get_memory_stats()

            graph_state = {}
            if self.l3_graph and self.l3_graph.previous_graph is not None:
                graph = self.l3_graph.previous_graph
                graph_state = {
                    "num_nodes": graph.num_nodes,
                    "num_edges": graph.edge_index.size(1)
                    if hasattr(graph, "edge_index")
                    else 0,
                    "has_features": hasattr(graph, "x") and graph.x is not None,
                }

            return {
                "memory": memory_stats,
                "graph": graph_state,
                "synchronized": True,  # Since we use unified method
            }

        except Exception as e:
            logger.error(f"Failed to get memory/graph state: {e}")
            return {"memory": {}, "graph": {}, "synchronized": False, "error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "initialized": self._initialized,
            "total_cycles": self.cycle_count,
            "memory_stats": self.l2_memory.get_memory_stats(),
            "reasoning_history_length": len(self.reasoning_history),
            "average_quality": np.mean([h["quality"] for h in self.reasoning_history])
            if self.reasoning_history
            else 0.0,
        }

    def get_insights(self, limit: int = 5) -> Dict[str, Any]:
        """Get recent insights discovered by the agent."""
        try:
            from ...detection.insight_registry import InsightFactRegistry

            registry = InsightFactRegistry()

            # Get insights from registry
            insights = registry.get_recent_insights(limit=limit)

            # Add metadata
            return {
                "total_insights": registry.total_insights,
                "recent_insights": [
                    {
                        "question": i.question,
                        "answer": i.answer,
                        "timestamp": i.timestamp,
                        "context": i.context,
                        "importance": i.importance,
                    }
                    for i in insights
                ],
                "categories": registry.get_categories(),
            }
        except Exception as e:
            logger.error(f"Failed to get insights: {e}")
            return {"total_insights": 0, "recent_insights": [], "categories": []}

    def search_insights(self, concept: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for insights related to a concept."""
        try:
            from ...detection.insight_registry import InsightFactRegistry

            registry = InsightFactRegistry()

            # Search insights
            results = registry.search_insights(concept, limit=limit)

            return [
                {
                    "question": i.question,
                    "answer": i.answer,
                    "relevance": i.relevance_score
                    if hasattr(i, "relevance_score")
                    else 1.0,
                    "timestamp": i.timestamp,
                    "importance": i.importance,
                }
                for i in results
            ]
        except Exception as e:
            logger.error(f"Failed to search insights: {e}")
            return []

    def save_state(self) -> bool:
        """Save agent state (memory and graph) using DataStore."""
        if not self.datastore:
            logger.warning("No DataStore configured, falling back to legacy save")
            return self._legacy_save_state()

        try:
            # Collect episodes from L2 memory
            if self.l2_memory and hasattr(self.l2_memory, "episodes"):
                episodes_to_save = []
                for episode in self.l2_memory.episodes:
                    # Convert episode to dict format for DataStore
                    episode_dict = {
                        "text": episode.text,
                        "vec": episode.vec,
                        "c_value": getattr(
                            episode, "c", 0.5
                        ),  # Episode uses 'c' attribute
                        "timestamp": getattr(episode, "timestamp", time.time()),
                    }
                    episodes_to_save.append(episode_dict)

                # Save episodes via DataStore
                self.datastore.save_episodes(episodes_to_save, namespace="agent_state")
                logger.info(f"Saved {len(episodes_to_save)} episodes via DataStore")

            # Save graph from L3
            if (
                self.l3_graph
                and hasattr(self.l3_graph, "previous_graph")
                and self.l3_graph.previous_graph is not None
            ):
                self.datastore.save_graph(
                    self.l3_graph.previous_graph,
                    graph_id="main_graph",
                    namespace="agent_state",
                )
                logger.info("Saved graph via DataStore")

            return True

        except Exception as e:
            logger.error(f"Failed to save agent state via DataStore: {e}")
            return False

    def _legacy_save_state(self) -> bool:
        """Legacy save method - to be deprecated."""
        try:
            success = True

            # Save L2 memory
            if self.l2_memory:
                memory_saved = self.l2_memory.save()
                if not memory_saved:
                    logger.warning("Failed to save L2 memory")
                    success = False
                else:
                    logger.info("L2 memory saved successfully")

            # Save L3 graph
            if self.l3_graph and self.l3_graph.previous_graph is not None:
                try:
                    self.l3_graph.save_graph(self.l3_graph.previous_graph)
                    logger.info("L3 graph saved successfully")
                except Exception as e:
                    logger.warning(f"Failed to save L3 graph: {e}")
                    success = False

            return success

        except Exception as e:
            logger.error(f"Failed to save agent state: {e}")
            return False

    def load_state(self) -> bool:
        """Load agent state (memory and graph) using DataStore."""
        if not self.datastore:
            logger.warning("No DataStore configured, falling back to legacy load")
            return self._legacy_load_state()

        try:
            # Load episodes into L2 memory
            if self.l2_memory:
                loaded_episodes = self.datastore.load_episodes(namespace="agent_state")
                if loaded_episodes:
                    # Clear existing episodes and load new ones
                    self.l2_memory.episodes = []
                    for ep_dict in loaded_episodes:
                        # Create episode object from dict
                        episode = Episode(
                            text=ep_dict["text"],
                            vec=ep_dict["vec"],
                            c=ep_dict.get(
                                "c_value", 0.5
                            ),  # Note: Episode uses 'c' not 'c_value'
                            timestamp=ep_dict.get("timestamp", time.time()),
                        )
                        self.l2_memory.episodes.append(episode)

                    logger.info(f"Loaded {len(loaded_episodes)} episodes via DataStore")
                else:
                    logger.warning("No episodes found in DataStore")

            # Load graph into L3
            if self.l3_graph:
                loaded_graph = self.datastore.load_graph(
                    graph_id="main_graph", namespace="agent_state"
                )
                if loaded_graph is not None:
                    self.l3_graph.previous_graph = loaded_graph
                    logger.info(
                        f"Loaded graph via DataStore: {loaded_graph.num_nodes} nodes"
                    )
                else:
                    logger.warning("No graph found in DataStore")

            return True

        except Exception as e:
            logger.error(f"Failed to load agent state via DataStore: {e}")
            return False

    def _legacy_load_state(self) -> bool:
        """Legacy load method - to be deprecated."""
        try:
            success = True

            # Load L2 memory
            if self.l2_memory:
                memory_loaded = self.l2_memory.load()
                if memory_loaded:
                    logger.info(
                        f"L2 memory loaded: {len(self.l2_memory.episodes)} episodes"
                    )
                else:
                    logger.warning("No existing L2 memory found")
                    success = False

            # Load L3 graph
            if self.l3_graph:
                loaded_graph = self.l3_graph.load_graph()
                if loaded_graph is not None:
                    self.l3_graph.previous_graph = loaded_graph
                    logger.info(f"L3 graph loaded: {loaded_graph.num_nodes} nodes")
                else:
                    logger.warning("No existing L3 graph found")
                    success = False

            return success

        except Exception as e:
            logger.error(f"Failed to load agent state: {e}")
            return False


# Backward compatibility function
def cycle(memory, question: str, previous_graph=None, **kwargs) -> Dict[str, Any]:
    """
    Backward compatible cycle function.

    Note: This creates a new agent each time, which is not optimal.
    Consider using MainAgent directly for better performance.
    """
    try:
        # Create temporary agent
        agent = MainAgent()

        # If memory is provided, try to extract documents
        if hasattr(memory, "episodes") and memory.episodes:
            for episode in memory.episodes:
                agent.add_document(episode.text, episode.c)

        # Process question
        result = agent.process_question(question, max_cycles=3, verbose=False)

        # Return in old format
        return {
            "answer": result.get("response", ""),
            "documents": result.get("documents", []),
            "graph": result.get("graph"),
            "metrics": result.get("metrics", {}),
            "success": result.get("success", True),
        }

    except Exception as e:
        logger.error(f"Legacy cycle function failed: {e}")
        return {
            "answer": f"Error: {e}",
            "documents": [],
            "graph": None,
            "metrics": {},
            "success": False,
        }


__all__ = ["MainAgent", "CycleResult", "cycle"]

# Import configurable agent for easy migration
try:
    from .configurable_agent import AgentConfig, AgentMode, ConfigurableAgent

    __all__.extend(["ConfigurableAgent", "AgentConfig", "AgentMode"])
except ImportError:
    pass  # Configurable agent not available yet
