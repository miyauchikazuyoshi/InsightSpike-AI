"""
Fixed Exploration Loop - Properly handles graph comparison for spike detection
"""

import logging
from typing import Dict, List, Any, Optional

from .interfaces import ExplorationParams, ExplorationResult

logger = logging.getLogger(__name__)


class ExplorationLoopFixed:
    """
    Fixed exploration loop that properly compares graphs for spike detection.
    
    Key fix: Compare the current memory state (with temporary episode) against
    the state before ANY modifications in this exploration cycle.
    """
    
    def __init__(
        self,
        l1_monitor,
        l2_memory,
        l3_graph,
        spike_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize exploration loop with layer components.
        
        Args:
            l1_monitor: Layer 1 error monitor
            l2_memory: Layer 2 memory manager
            l3_graph: Layer 3 graph reasoner
            spike_thresholds: Optional override for spike detection thresholds
        """
        self.l1_monitor = l1_monitor
        self.l2_memory = l2_memory
        self.l3_graph = l3_graph
        
        # Default spike thresholds (can be overridden)
        self.spike_thresholds = spike_thresholds or {
            "delta_ged": -0.5,  # Significant structural change (simplification)
            "delta_ig": 0.2     # Information gain
        }
        
    def explore_once(
        self, 
        question: str, 
        params: ExplorationParams,
        previous_stable_graph: Any = None
    ) -> ExplorationResult:
        """
        Execute a single L1→L2→L3 exploration attempt.
        
        Key change: Uses previous_stable_graph for comparison instead of
        the graph state at the beginning of this exploration.
        
        Args:
            question: The query to process
            params: Exploration parameters
            previous_stable_graph: The stable graph from before this exploration cycle
            
        Returns:
            ExplorationResult with spike detection status
        """
        logger.debug(
            f"Exploration attempt {params.attempt_number} with "
            f"radius={params.radius:.2f}, topK={params.topk_l2}"
        )
        
        # Layer 1: Error analysis with exploration radius
        l1_analysis = self._run_layer1(question, params.radius)
        
        # Layer 2: Memory retrieval with adaptive topK
        retrieved_docs = self._run_layer2(
            question, 
            params.topk_l2, 
            params.radius
        )
        
        # Create synthetic episode from question and retrieved context
        # This simulates what would happen if we added this knowledge
        synthetic_text = self._create_synthetic_episode(question, retrieved_docs)
        
        # Build current state with synthetic episode
        docs_for_graph = self._prepare_documents_for_graph(
            retrieved_docs, 
            synthetic_text
        )
        
        # Layer 3: Graph analysis and spike detection
        # Compare against the stable graph from before exploration
        context = {
            "l1_analysis": l1_analysis,
            "question": question,
            "previous_graph": previous_stable_graph
        }
        
        graph_analysis = self._run_layer3(docs_for_graph, context)
        
        # Check for spike
        spike_detected = self._check_spike(graph_analysis)
        confidence = graph_analysis.get("reasoning_quality", 0.0)
        
        if spike_detected:
            logger.info(
                f"🎯 Spike detected at attempt {params.attempt_number}! "
                f"ΔGED={graph_analysis.get('metrics', {}).get('delta_ged', 0):.3f}, "
                f"ΔIG={graph_analysis.get('metrics', {}).get('delta_ig', 0):.3f}"
            )
        
        # Build result
        metrics = {
            "l1_uncertainty": l1_analysis.get("uncertainty", 1.0),
            "l1_known_ratio": len(l1_analysis.get("known_elements", [])) / 
                             max(1, len(l1_analysis.get("known_elements", [])) + 
                                 len(l1_analysis.get("unknown_elements", []))),
            "l2_retrieval_count": len(retrieved_docs),
            "l3_delta_ged": graph_analysis.get("metrics", {}).get("delta_ged", 0),
            "l3_delta_ig": graph_analysis.get("metrics", {}).get("delta_ig", 0),
            "l3_confidence": confidence
        }
        
        return ExplorationResult(
            spike_detected=spike_detected,
            confidence=confidence,
            retrieved_docs=retrieved_docs,
            graph_analysis=graph_analysis,
            metrics=metrics,
            params=params,
            synthetic_episode=synthetic_text  # Include for potential addition
        )
    
    def _create_synthetic_episode(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Create synthetic episode text from question and retrieved documents.
        This represents the potential new knowledge to be added.
        """
        # Combine question with retrieved context
        context_texts = []
        for doc in retrieved_docs[:3]:  # Use top 3 documents
            if isinstance(doc, dict):
                text = doc.get("text", "")
            else:
                text = str(doc)
            if text:
                context_texts.append(str(text))
        
        # Create a synthetic episode that represents the integration
        # of the question with the retrieved knowledge
        if context_texts:
            combined_text = f"Q: {question}\nContext: {' '.join(context_texts)}"
        else:
            combined_text = f"Q: {question}"
            
        return combined_text
    
    def _prepare_documents_for_graph(
        self, 
        retrieved_docs: List[Dict[str, Any]], 
        synthetic_text: str
    ) -> List[Dict[str, Any]]:
        """
        Prepare documents for graph building, including synthetic episode.
        """
        # Get all current episodes from memory
        docs_for_graph = []
        
        if hasattr(self.l2_memory, 'episodes'):
            for i, ep in enumerate(self.l2_memory.episodes):
                docs_for_graph.append({
                    "text": ep.text,
                    "embedding": ep.vec if hasattr(ep, 'vec') else None,
                    "episode_idx": i,
                    "is_synthetic": False
                })
        
        # Add synthetic episode
        if synthetic_text:
            # Generate embedding for synthetic text
            synthetic_embedding = None
            if hasattr(self.l2_memory, 'embedder'):
                try:
                    synthetic_embedding = self.l2_memory.embedder.encode_single(synthetic_text)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for synthetic episode: {e}")
            
            docs_for_graph.append({
                "text": synthetic_text,
                "embedding": synthetic_embedding,
                "episode_idx": len(docs_for_graph),
                "is_synthetic": True
            })
        
        return docs_for_graph
    
    def _run_layer1(self, question: str, radius: float) -> Dict[str, Any]:
        """Run Layer 1 error analysis"""
        try:
            if hasattr(self.l1_monitor, 'analyze_uncertainty'):
                try:
                    return self.l1_monitor.analyze_uncertainty(
                        question,
                        exploration_radius=radius
                    )
                except TypeError:
                    return self.l1_monitor.analyze_uncertainty(question)
            else:
                return {"uncertainty": 0.5, "known_elements": [], "unknown_elements": []}
                
        except Exception as e:
            logger.warning(f"L1 analysis failed: {e}")
            return {"uncertainty": 1.0, "known_elements": [], "unknown_elements": []}
    
    def _run_layer2(
        self, 
        question: str, 
        topk: int, 
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Run Layer 2 memory retrieval"""
        try:
            # First, encode the question text to get embedding vector
            if hasattr(self.l2_memory, '_encode_text'):
                query_vec = self.l2_memory._encode_text(question)
            elif hasattr(self.l2_memory, 'embedder'):
                query_vec = self.l2_memory.embedder.encode_single(question)
            else:
                logger.warning("L2 memory has no embedding capability")
                return []
            
            # Search for similar episodes
            if hasattr(self.l2_memory, 'search'):
                results = self.l2_memory.search(query_vec, k=topk, min_score=similarity_threshold)
                
                # Format results
                formatted_results = []
                for item in results:
                    if isinstance(item, tuple) and len(item) >= 3:
                        idx, score, episode = item
                        formatted_results.append({
                            "text": episode.text,
                            "score": score,
                            "episode": episode,
                            "embedding": episode.vec if hasattr(episode, 'vec') else None,
                            "index": idx
                        })
                    elif isinstance(item, dict):
                        formatted_results.append(item)
                        
                return formatted_results
            else:
                logger.warning("L2 memory has no search method")
                return []
                
        except Exception as e:
            logger.warning(f"L2 retrieval failed: {e}")
            return []
    
    def _run_layer3(
        self, 
        documents: List[Dict[str, Any]], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run Layer 3 graph analysis"""
        try:
            if not self.l3_graph:
                return {"reasoning_quality": 0.5, "metrics": {}}
                
            return self.l3_graph.analyze_documents(documents, context)
            
        except Exception as e:
            logger.warning(f"L3 analysis failed: {e}")
            return {"reasoning_quality": 0.0, "metrics": {}}
    
    def _check_spike(self, graph_analysis: Dict[str, Any]) -> bool:
        """
        Check if spike conditions are met.
        
        A spike is detected when:
        - ΔGED ≤ threshold (significant structural change)
        - ΔIG ≥ threshold (information gain)
        """
        # First check if spike_detected is already set
        if graph_analysis.get("spike_detected", False):
            return True
            
        # Otherwise check metrics
        metrics = graph_analysis.get("metrics", {})
        delta_ged = metrics.get("delta_ged", 0)
        delta_ig = metrics.get("delta_ig", 0)
        
        spike_detected = (
            delta_ged <= self.spike_thresholds["delta_ged"] and
            delta_ig >= self.spike_thresholds["delta_ig"]
        )
        
        return spike_detected