"""
Exploration Loop - Manages L1-L2-L3 iterations without LLM
"""

import logging
from typing import Dict, List, Any, Optional

from .interfaces import ExplorationParams, ExplorationResult
from ...config.normalized import NormalizedConfig
from ...config import get_config

logger = logging.getLogger(__name__)


class ExplorationLoop:
    """
    Manages the L1→L2→L3 exploration loop.
    
    This component executes exploration attempts without calling the LLM,
    checking for spikes through graph analysis.
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
            "delta_ged": -0.5,  # Significant structural change
            "delta_ig": 0.2     # Information gain
        }
        
    def explore_once(
        self, 
        question: str, 
        params: ExplorationParams,
        add_temporary_episode: bool = True
    ) -> ExplorationResult:
        """
        Execute a single L1→L2→L3 exploration attempt.
        
        Args:
            question: The query to process
            params: Exploration parameters
            add_temporary_episode: Whether to add a temporary episode for spike detection
            
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
        
        # Get the graph state BEFORE adding temporary episode
        # This is crucial for proper spike detection
        if hasattr(self.l3_graph, 'previous_graph'):
            graph_before_temp = self.l3_graph.previous_graph
        else:
            graph_before_temp = None
        
        # Add temporary episode for spike detection if requested
        temp_episode_idx = None
        if add_temporary_episode:
            # Create a temporary episode from retrieved context
            temp_text = self._create_temp_episode_text(question, retrieved_docs)
            if temp_text and len(temp_text) > len(question) + 20:  # Ensure we have meaningful content
                temp_episode_idx = self._add_temporary_episode(temp_text, params.temperature)
        
        # Layer 3: Graph analysis and spike detection
        graph_analysis = {}
        spike_detected = False
        confidence = 0.0
        
        if retrieved_docs or temp_episode_idx is not None:
            # Build documents list including temporary episode if added
            if temp_episode_idx is not None and hasattr(self.l2_memory, 'episodes'):
                # Get all current episodes including the temporary one
                all_episodes = self.l2_memory.episodes
                docs_for_graph = []
                
                # Convert episodes to document format
                for i, ep in enumerate(all_episodes):
                    docs_for_graph.append({
                        "text": ep.text,
                        "embedding": ep.vec,
                        "episode_idx": i,
                        "is_temporary": i == temp_episode_idx
                    })
            else:
                # No temporary episode, use retrieved docs
                docs_for_graph = self._ensure_embeddings(retrieved_docs)
            
            # Pass the graph state from before temporary episode was added
            # Derive NormSpec from config to keep Layer norms consistent across layers
            try:
                _nc = NormalizedConfig.from_any(get_config())
                _norm_spec = _nc.norm_spec
            except Exception:
                _norm_spec = None
            context = {
                "l1_analysis": l1_analysis,
                "question": question,
                "previous_graph": graph_before_temp,  # Use pre-temp graph for comparison
                "norm_spec": _norm_spec,
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
        
        # Remove temporary episode if no spike was detected
        if temp_episode_idx is not None and not spike_detected:
            self._remove_temporary_episode(temp_episode_idx)
        
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
            params=params
        )
    
    def _run_layer1(self, question: str, radius: float) -> Dict[str, Any]:
        """Run Layer 1 error analysis"""
        try:
            # Check if L1 supports exploration radius
            if hasattr(self.l1_monitor, 'analyze_uncertainty'):
                # Try with exploration params if supported
                try:
                    return self.l1_monitor.analyze_uncertainty(
                        question,
                        exploration_radius=radius
                    )
                except TypeError:
                    # Fallback to basic call
                    return self.l1_monitor.analyze_uncertainty(question)
            else:
                # Fallback for older implementations
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
            
            # Try different method signatures for compatibility
            if hasattr(self.l2_memory, 'search_episodes'):
                # search_episodes expects numpy array
                try:
                    results = self.l2_memory.search_episodes(query_vec, top_k=topk)
                except TypeError:
                    # Fallback to simpler call
                    try:
                        results = self.l2_memory.search_episodes(query_vec)
                        # Limit results to topk
                        if results and len(results) > topk:
                            results = results[:topk]
                    except Exception as e:
                        logger.warning(f"search_episodes failed: {e}")
                        results = []
                        
                # Convert to expected format if needed
                if results:
                    formatted_results = []
                    for r in results:
                        if isinstance(r, tuple) and len(r) >= 2:
                            # Check if first element is Episode object
                            if hasattr(r[0], 'text'):
                                # (Episode, score) format from CachedMemoryManager
                                formatted_results.append({
                                    "text": r[0].text,
                                    "score": r[1],
                                    "episode": r[0],
                                    "embedding": r[0].vec if hasattr(r[0], 'vec') else None
                                })
                            elif len(r) >= 3 and hasattr(r[2], 'text'):
                                # (index, score, episode) format from search_episodes
                                formatted_results.append({
                                    "text": r[2].text,
                                    "score": r[1],
                                    "episode": r[2],
                                    "embedding": r[2].vec if hasattr(r[2], 'vec') else None
                                })
                            elif isinstance(r[0], str):
                                # (text, score) format
                                formatted_results.append({
                                    "text": r[0],
                                    "score": r[1],
                                    "episode": r[2] if len(r) > 2 else None
                                })
                            elif isinstance(r[0], int):
                                # Try to get episode from memory by index
                                if hasattr(self.l2_memory, 'episodes') and 0 <= r[0] < len(self.l2_memory.episodes):
                                    ep = self.l2_memory.episodes[r[0]]
                                    formatted_results.append({
                                        "text": ep.text,
                                        "score": r[1],
                                        "episode": ep,
                                        "embedding": ep.vec if hasattr(ep, 'vec') else None
                                    })
                        elif isinstance(r, dict):
                            formatted_results.append(r)
                        else:
                            # Try to extract text from episode object
                            formatted_results.append({
                                "text": getattr(r, 'text', str(r)),
                                "score": getattr(r, 'score', 0.5),
                                "embedding": getattr(r, 'vec', None)
                            })
                    return formatted_results
                return []
            else:
                logger.warning("L2 memory has no search_episodes method")
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
    
    def _create_temp_episode_text(self, question: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Create temporary episode text from question and retrieved documents.
        """
        # Combine question with retrieved context
        context_texts = []
        for doc in retrieved_docs[:3]:  # Use top 3 documents
            if isinstance(doc, dict):
                text = doc.get("text", "")
            else:
                text = str(doc)
            # Ensure text is string
            if text:
                context_texts.append(str(text))
        
        # Join with string conversion to be safe
        combined_text = f"Q: {question}\nContext: {' '.join(str(t) for t in context_texts)}"
        return combined_text
    
    def _add_temporary_episode(self, text: str, temperature: float = 1.0) -> Optional[int]:
        """
        Add a temporary episode to memory for spike detection.
        """
        try:
            # Calculate c_value based on temperature (exploration confidence)
            c_value = min(0.9, 0.5 + (1.0 - temperature) * 0.4)
            
            # Add episode with temporary flag
            metadata = {"temporary": True, "exploration": True}
            episode_idx = self.l2_memory.add_episode(
                text,
                c_value=c_value,
                metadata=metadata
            )
            
            logger.debug(f"Added temporary episode {episode_idx} with c_value={c_value}")
            return episode_idx
            
        except Exception as e:
            logger.warning(f"Failed to add temporary episode: {e}")
            return None
    
    def _remove_temporary_episode(self, episode_idx: int):
        """
        Remove a temporary episode from memory.
        """
        try:
            # Check if L2 memory supports episode removal
            if hasattr(self.l2_memory, 'remove_episode'):
                self.l2_memory.remove_episode(episode_idx)
                logger.debug(f"Removed temporary episode {episode_idx}")
            else:
                # If removal not supported, mark it for cleanup
                if hasattr(self.l2_memory, 'episodes') and 0 <= episode_idx < len(self.l2_memory.episodes):
                    self.l2_memory.episodes[episode_idx].metadata["to_remove"] = True
                    logger.debug(f"Marked temporary episode {episode_idx} for removal")
                    
        except Exception as e:
            logger.warning(f"Failed to remove temporary episode: {e}")
    
    def _ensure_embeddings(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensure all documents have embeddings for graph building.
        """
        docs_with_embeddings = []
        
        for doc in documents:
            if "embedding" in doc and doc["embedding"] is not None:
                # Already has embedding
                docs_with_embeddings.append(doc)
            elif "episode" in doc and hasattr(doc["episode"], "vec"):
                # Get embedding from episode
                doc_copy = doc.copy()
                doc_copy["embedding"] = doc["episode"].vec
                docs_with_embeddings.append(doc_copy)
            elif "text" in doc:
                # Generate embedding for text
                if hasattr(self.l2_memory, 'embedder'):
                    try:
                        embedding = self.l2_memory.embedder.encode_single(doc["text"])
                        doc_copy = doc.copy()
                        doc_copy["embedding"] = embedding
                        docs_with_embeddings.append(doc_copy)
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding: {e}")
                        docs_with_embeddings.append(doc)
                else:
                    docs_with_embeddings.append(doc)
            else:
                # No text or embedding available
                docs_with_embeddings.append(doc)
                
        return docs_with_embeddings
