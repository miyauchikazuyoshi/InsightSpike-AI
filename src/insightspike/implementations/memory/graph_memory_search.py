"""
Graph-Based Memory Search
========================

Implements multi-hop graph traversal for associative memory retrieval.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class GraphMemorySearch:
    """
    Enhanced memory search using graph traversal.
    
    Features:
    - Multi-hop neighbor exploration
    - Path-based relevance scoring
    - Concept chaining through graph edges
    - Subgraph extraction for context
    """
    
    def __init__(self, config=None):
        self.config = config
        self.hop_limit = 2  # Default 2-hop traversal
        self.neighbor_threshold = 0.4  # Min similarity for neighbor inclusion
        self.path_decay = 0.7  # Relevance decay per hop
        
        if config:
            # Get settings from config
            if hasattr(config, "graph"):
                self.hop_limit = getattr(config.graph, "hop_limit", 2)
                self.neighbor_threshold = getattr(config.graph, "neighbor_threshold", 0.4)
                self.path_decay = getattr(config.graph, "path_decay", 0.7)
    
    def search_with_graph(
        self,
        query_embedding: np.ndarray,
        episodes: List[Any],
        graph_data: Optional[Any] = None,
        k: int = 10,
        enable_multi_hop: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search memory using graph-based traversal.
        
        Args:
            query_embedding: Query vector
            episodes: List of memory episodes
            graph_data: Graph structure (PyG Data object)
            k: Number of results to return
            enable_multi_hop: Whether to use multi-hop search
            
        Returns:
            List of relevant documents with graph-enhanced scoring
        """
        if not episodes:
            return []
        
        # Start with direct similarity search
        direct_results = self._direct_similarity_search(
            query_embedding, episodes, k=k*2  # Get more candidates
        )
        
        if not enable_multi_hop or graph_data is None:
            return direct_results[:k]
        
        # Enhance with graph traversal
        enhanced_results = self._multi_hop_search(
            query_embedding, episodes, direct_results, graph_data, k
        )
        
        return enhanced_results
    
    def _direct_similarity_search(
        self,
        query_embedding: np.ndarray,
        episodes: List[Any],
        k: int
    ) -> List[Dict[str, Any]]:
        """Standard cosine similarity search."""
        # Extract embeddings from episodes
        embeddings = []
        valid_episodes = []
        
        for ep in episodes:
            if hasattr(ep, 'vec') and ep.vec is not None:
                embeddings.append(ep.vec)
                valid_episodes.append(ep)
        
        if not embeddings:
            return []
        
        # Calculate similarities
        embeddings_matrix = np.array(embeddings)
        query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]
        
        # Sort by similarity
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            episode = valid_episodes[idx]
            results.append({
                "text": episode.text,
                "similarity": float(similarities[idx]),
                "index": episodes.index(episode),
                "c_value": getattr(episode, 'c_value', 0.5),
                "hop": 0,  # Direct match
                "path": [episodes.index(episode)]
            })
        
        return results
    
    def _multi_hop_search(
        self,
        query_embedding: np.ndarray,
        episodes: List[Any],
        direct_results: List[Dict[str, Any]],
        graph_data: Any,
        k: int
    ) -> List[Dict[str, Any]]:
        """
        Enhance search results using multi-hop graph traversal.
        """
        # Track visited nodes to avoid cycles
        visited = set()
        enhanced_results = []
        
        # Use top direct results as starting points
        start_nodes = [r["index"] for r in direct_results[:5]]
        
        for start_idx in start_nodes:
            # Explore neighbors up to hop_limit
            hop_results = self._explore_neighbors(
                start_idx, query_embedding, episodes, graph_data, 
                visited, current_hop=0, max_hops=self.hop_limit
            )
            enhanced_results.extend(hop_results)
        
        # Combine with direct results
        all_results = direct_results + enhanced_results
        
        # Remove duplicates and re-rank
        unique_results = self._deduplicate_results(all_results)
        
        # Apply graph-based re-ranking
        reranked = self._rerank_with_graph_context(unique_results, graph_data)
        
        return reranked[:k]
    
    def _explore_neighbors(
        self,
        node_idx: int,
        query_embedding: np.ndarray,
        episodes: List[Any],
        graph_data: Any,
        visited: Set[int],
        current_hop: int,
        max_hops: int,
        path: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Recursively explore graph neighbors.
        """
        if current_hop > max_hops or node_idx in visited:
            return []
        
        visited.add(node_idx)
        path = path or []
        path = path + [node_idx]
        
        results = []
        
        # Get neighbors from graph
        neighbors = self._get_neighbors(node_idx, graph_data)
        
        for neighbor_idx in neighbors:
            if neighbor_idx >= len(episodes) or neighbor_idx in visited:
                continue
            
            episode = episodes[neighbor_idx]
            if not hasattr(episode, 'vec') or episode.vec is None:
                continue
            
            # Calculate relevance with path decay
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                episode.vec.reshape(1, -1)
            )[0, 0]
            
            # Apply decay based on hop distance
            decayed_similarity = similarity * (self.path_decay ** (current_hop + 1))
            
            if decayed_similarity > self.neighbor_threshold:
                results.append({
                    "text": episode.text,
                    "similarity": float(decayed_similarity),
                    "index": neighbor_idx,
                    "c_value": getattr(episode, 'c_value', 0.5),
                    "hop": current_hop + 1,
                    "path": path + [neighbor_idx]
                })
                
                # Recursive exploration
                if current_hop + 1 < max_hops:
                    sub_results = self._explore_neighbors(
                        neighbor_idx, query_embedding, episodes, graph_data,
                        visited, current_hop + 1, max_hops, path
                    )
                    results.extend(sub_results)
        
        return results
    
    def _get_neighbors(self, node_idx: int, graph_data: Any) -> List[int]:
        """Extract neighbors of a node from the graph."""
        if graph_data is None or not hasattr(graph_data, 'edge_index'):
            return []
        
        edge_index = graph_data.edge_index
        
        # Find edges where node_idx is the source
        neighbors = []
        for i in range(edge_index.size(1)):
            if edge_index[0, i].item() == node_idx:
                neighbors.append(edge_index[1, i].item())
        
        return list(set(neighbors))  # Remove duplicates
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results, keeping the highest scoring version."""
        seen = {}
        
        for result in results:
            idx = result["index"]
            if idx not in seen or result["similarity"] > seen[idx]["similarity"]:
                seen[idx] = result
        
        return list(seen.values())
    
    def _rerank_with_graph_context(
        self,
        results: List[Dict[str, Any]],
        graph_data: Any
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results considering graph structure and paths.
        """
        # Calculate graph-based features for re-ranking
        for result in results:
            # Bonus for direct matches
            if result["hop"] == 0:
                result["graph_score"] = result["similarity"] * 1.2
            else:
                # Consider path length and connectivity
                path_length_penalty = 1.0 / (1 + result["hop"])
                result["graph_score"] = result["similarity"] * path_length_penalty
            
            # Boost highly connected nodes (hubs)
            node_degree = self._get_node_degree(result["index"], graph_data)
            connectivity_bonus = min(0.1, node_degree * 0.01)
            result["graph_score"] += connectivity_bonus
        
        # Sort by graph score
        results.sort(key=lambda x: x["graph_score"], reverse=True)
        
        return results
    
    def _get_node_degree(self, node_idx: int, graph_data: Any) -> int:
        """Get the degree (number of connections) of a node."""
        if graph_data is None or not hasattr(graph_data, 'edge_index'):
            return 0
        
        edge_index = graph_data.edge_index
        degree = 0
        
        for i in range(edge_index.size(1)):
            if edge_index[0, i].item() == node_idx or edge_index[1, i].item() == node_idx:
                degree += 1
        
        return degree
    
    def extract_subgraph(
        self,
        center_nodes: List[int],
        graph_data: Any,
        radius: int = 1
    ) -> Dict[str, Any]:
        """
        Extract a subgraph around the given nodes.
        
        Useful for providing local context to the LLM.
        """
        if graph_data is None:
            return {"nodes": center_nodes, "edges": [], "radius": radius}
        
        # Collect nodes within radius
        subgraph_nodes = set(center_nodes)
        current_layer = set(center_nodes)
        
        for _ in range(radius):
            next_layer = set()
            for node in current_layer:
                neighbors = self._get_neighbors(node, graph_data)
                next_layer.update(neighbors)
            
            subgraph_nodes.update(next_layer)
            current_layer = next_layer
        
        # Extract edges within subgraph
        subgraph_edges = []
        edge_index = graph_data.edge_index
        
        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            if src in subgraph_nodes and dst in subgraph_nodes:
                subgraph_edges.append((src, dst))
        
        return {
            "nodes": list(subgraph_nodes),
            "edges": subgraph_edges,
            "radius": radius,
            "center_nodes": center_nodes
        }