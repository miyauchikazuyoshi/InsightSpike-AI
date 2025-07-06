"""
L2 Enhanced Scalable Memory Manager with Conflict-Based Splitting
================================================================

Extends L2MemoryManager with scalable graph features and conflict detection.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from .layer2_memory_manager import L2MemoryManager, Episode
from ..learning.scalable_graph_manager import ScalableGraphManager
from ...config import get_config
from ...utils.embedder import get_model

logger = logging.getLogger(__name__)


class L2EnhancedScalableMemory(L2MemoryManager):
    """
    Enhanced memory manager with scalable graph operations and conflict-based splitting.
    
    Features:
    - All features from L2MemoryManager
    - FAISS-based scalable graph management
    - Conflict detection and automatic episode splitting
    - Dynamic graph-based importance calculation
    - Performance optimizations for large-scale operations
    """
    
    def __init__(
        self,
        dim: int = None,
        config=None,
        use_scalable_graph: bool = True,
        conflict_split_threshold: int = 2
    ):
        # Initialize parent class
        super().__init__(dim, config)
        
        self.use_scalable_graph = use_scalable_graph
        self.conflict_split_threshold = conflict_split_threshold
        
        # Initialize scalable graph manager if enabled
        if self.use_scalable_graph:
            self.scalable_graph = ScalableGraphManager(
                embedding_dim=self.dim,
                similarity_threshold=self.config.reasoning.similarity_threshold,
                top_k=getattr(self.config.reasoning, 'graph_top_k', 50),
                conflict_threshold=getattr(self.config.reasoning, 'conflict_threshold', 0.8)
            )
            logger.info("Initialized ScalableGraphManager for enhanced memory operations")
        else:
            self.scalable_graph = None
        
        # Track conflicts for splitting decisions
        self.recent_conflicts: List[Dict[str, Any]] = []
        
    def store_episode(
        self, text: str, c_value: float = 0.5, metadata: Optional[Dict] = None
    ) -> bool:
        """Store episode with conflict detection and potential splitting."""
        try:
            # First, use parent's store_episode logic
            result = super().store_episode(text, c_value, metadata)
            
            if not result:
                return False
            
            # If using scalable graph, perform additional operations
            if self.use_scalable_graph and self.episodes:
                latest_episode = self.episodes[-1]
                latest_idx = len(self.episodes) - 1
                
                # Add to scalable graph and check for conflicts
                graph_result = self.scalable_graph.add_episode_node(
                    embedding=latest_episode.vec,
                    index=latest_idx,
                    metadata={
                        "text": latest_episode.text,
                        "c_value": latest_episode.c,
                        "timestamp": metadata.get("timestamp") if metadata else None
                    }
                )
                
                if graph_result["success"]:
                    # Check if conflicts warrant splitting
                    conflicts = graph_result.get("conflicts", [])
                    if conflicts:
                        self.recent_conflicts.extend(conflicts)
                        logger.info(f"Detected {len(conflicts)} conflicts for episode {latest_idx}")
                        
                        # Check if we should split
                        if self.scalable_graph.should_split_episode(conflicts):
                            self._handle_conflict_split(latest_idx, conflicts)
                    
                    # Update episode importance based on graph
                    if "importance" in graph_result:
                        self._update_episode_importance(latest_idx, graph_result["importance"])
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store episode with enhanced features: {e}")
            return False
    
    def _handle_conflict_split(self, episode_idx: int, conflicts: List[Dict[str, Any]]):
        """Handle episode splitting based on detected conflicts."""
        try:
            # Get split candidates
            candidates = self.scalable_graph.get_split_candidates(episode_idx)
            
            if not candidates:
                logger.info("No suitable split candidates found")
                return
            
            # Perform split operation
            split_indices = self.split_by_conflict(episode_idx, candidates)
            
            if split_indices:
                logger.info(f"Split episode {episode_idx} into {len(split_indices)} new episodes")
                
                # Update scalable graph with new episodes
                if self.scalable_graph:
                    self.scalable_graph.update_from_episodes([
                        {
                            "embedding": ep.vec,
                            "text": ep.text,
                            "c_value": ep.c,
                            "timestamp": ep.metadata.get("timestamp")
                        }
                        for ep in self.episodes
                    ])
            
        except Exception as e:
            logger.error(f"Failed to handle conflict split: {e}")
    
    def split_by_conflict(
        self, 
        episode_index: int, 
        conflict_candidates: List[Tuple[int, float]]
    ) -> List[int]:
        """
        Split an episode based on conflicts with other episodes.
        
        Args:
            episode_index: Index of episode to potentially split
            conflict_candidates: List of (index, similarity) tuples for conflicting episodes
            
        Returns:
            List of new episode indices created from split
        """
        try:
            if not (0 <= episode_index < len(self.episodes)):
                logger.warning(f"Invalid episode index for conflict split: {episode_index}")
                return []
            
            episode = self.episodes[episode_index]
            
            # Analyze text for potential split points
            text_parts = self._analyze_text_for_splits(episode.text)
            
            if len(text_parts) <= 1:
                logger.info("Episode text cannot be meaningfully split")
                return []
            
            # Create new episodes from split parts
            new_episodes = []
            for i, part in enumerate(text_parts):
                # Re-encode each part
                model = get_model()
                part_vec = model.encode(
                    [part], convert_to_numpy=True, normalize_embeddings=True
                )[0]
                
                # Inherit C-value with slight reduction for splits
                split_c = episode.c * (0.9 - 0.1 * i)  # Decay C-value for later splits
                
                # Create metadata indicating split origin
                split_metadata = episode.metadata.copy() if episode.metadata else {}
                split_metadata["split_from"] = episode_index
                split_metadata["split_part"] = i + 1
                split_metadata["split_total"] = len(text_parts)
                split_metadata["split_reason"] = "conflict"
                
                new_episode = Episode(part_vec, part, split_c, split_metadata)
                new_episodes.append(new_episode)
            
            # Remove original episode
            del self.episodes[episode_index]
            
            # Add new episodes
            new_indices = []
            for new_ep in new_episodes:
                self.episodes.append(new_ep)
                new_indices.append(len(self.episodes) - 1)
            
            # Retrain index if needed
            if len(self.episodes) >= 2:
                self._train_index()
            
            logger.info(f"Split episode {episode_index} into {len(new_indices)} parts due to conflicts")
            return new_indices
            
        except Exception as e:
            logger.error(f"Failed to split episode by conflict: {e}")
            return []
    
    def _analyze_text_for_splits(self, text: str) -> List[str]:
        """Analyze text to find meaningful split points."""
        # Simple implementation: split by sentences or key phrases
        import re
        
        # Split by common delimiters
        delimiters = [
            r'\. ',  # Period followed by space
            r'! ',   # Exclamation
            r'\? ',  # Question mark
            r' \| ', # Pipe separator
            r'\n',   # Newline
        ]
        
        # Create regex pattern
        pattern = '|'.join(delimiters)
        parts = re.split(pattern, text)
        
        # Filter out empty parts and recombine very short ones
        meaningful_parts = []
        current_part = ""
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            if len(current_part) + len(part) < 50:  # Minimum meaningful length
                current_part += " " + part if current_part else part
            else:
                if current_part:
                    meaningful_parts.append(current_part)
                current_part = part
        
        if current_part:
            meaningful_parts.append(current_part)
        
        return meaningful_parts if len(meaningful_parts) > 1 else [text]
    
    def _update_episode_importance(self, episode_idx: int, importance: float):
        """Update episode C-value based on graph-derived importance."""
        if 0 <= episode_idx < len(self.episodes):
            episode = self.episodes[episode_idx]
            
            # Blend current C-value with graph importance
            # This maintains some stability while incorporating graph structure
            alpha = 0.3  # Weight for graph importance
            new_c = (1 - alpha) * episode.c + alpha * importance
            
            # Apply bounds
            new_c = max(self.c_min, min(self.c_max, new_c))
            
            episode.c = new_c
            logger.debug(f"Updated episode {episode_idx} C-value based on graph importance: {new_c:.3f}")
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph structure."""
        if not self.use_scalable_graph or not self.scalable_graph:
            return {"graph_enabled": False}
        
        graph = self.scalable_graph.graph
        return {
            "graph_enabled": True,
            "nodes": graph.num_nodes,
            "edges": graph.edge_index.size(1) // 2 if graph.edge_index.numel() > 0 else 0,
            "density": (graph.edge_index.size(1) / (graph.num_nodes * (graph.num_nodes - 1))) 
                      if graph.num_nodes > 1 else 0,
            "recent_conflicts": len(self.recent_conflicts),
            "faiss_index_size": len(self.scalable_graph.embeddings)
        }
    
    def search_episodes_with_graph(
        self, query: str, k: int = 5, min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Enhanced search that considers graph structure."""
        # First, get base results from parent class
        base_results = self.search_episodes(query, k * 2, min_similarity)
        
        if not self.use_scalable_graph or not self.scalable_graph or not base_results:
            return base_results[:k]
        
        # Enhance results with graph-based reranking
        enhanced_results = []
        
        for result in base_results:
            idx = result["index"]
            
            # Get node importance from graph
            importance = self.scalable_graph._calculate_node_importance(idx)
            
            # Create enhanced score combining similarity, C-value, and graph importance
            enhanced_score = (
                0.5 * result["weighted_score"] +  # Original weighted score
                0.3 * result["similarity"] +       # Pure similarity
                0.2 * importance                   # Graph importance
            )
            
            result["graph_importance"] = importance
            result["enhanced_score"] = enhanced_score
            enhanced_results.append(result)
        
        # Sort by enhanced score
        enhanced_results.sort(key=lambda x: x["enhanced_score"], reverse=True)
        
        return enhanced_results[:k]
    
    def save(self, path: Optional[Path] = None) -> bool:
        """Save memory state including scalable graph."""
        # First save base memory
        base_result = super().save(path)
        
        if not base_result:
            return False
        
        # Save scalable graph index if enabled
        if self.use_scalable_graph and self.scalable_graph:
            try:
                if path is None:
                    config = get_config()
                    index_path = Path(config.memory.index_file)
                    faiss_path = index_path.parent / "scalable_index.faiss"
                else:
                    faiss_path = Path(path).parent / "scalable_index.faiss"
                
                self.scalable_graph.save_index(str(faiss_path))
                logger.info(f"Saved scalable graph index to {faiss_path}")
                
            except Exception as e:
                logger.error(f"Failed to save scalable graph: {e}")
                # Don't fail entire save if graph save fails
        
        return True
    
    def load(self, path: Optional[Path] = None) -> bool:
        """Load memory state including scalable graph."""
        # First load base memory
        base_result = super().load(path)
        
        if not base_result:
            return False
        
        # Load scalable graph index if enabled
        if self.use_scalable_graph and self.scalable_graph:
            try:
                if path is None:
                    config = get_config()
                    index_path = Path(config.memory.index_file)
                    faiss_path = index_path.parent / "scalable_index.faiss"
                else:
                    faiss_path = Path(path).parent / "scalable_index.faiss"
                
                if faiss_path.exists():
                    self.scalable_graph.load_index(str(faiss_path))
                    logger.info(f"Loaded scalable graph index from {faiss_path}")
                    
                    # Rebuild graph structure from episodes
                    self.scalable_graph.update_from_episodes([
                        {
                            "embedding": ep.vec,
                            "text": ep.text,
                            "c_value": ep.c,
                            "timestamp": ep.metadata.get("timestamp")
                        }
                        for ep in self.episodes
                    ])
                
            except Exception as e:
                logger.error(f"Failed to load scalable graph: {e}")
                # Don't fail entire load if graph load fails
        
        return True


