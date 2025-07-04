#!/usr/bin/env python3
"""
Layer2 enhancement: Graph-informed episode integration
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


class GraphInformedIntegration:
    """
    Enhanced episode integration using graph structure from Layer3
    """
    
    def __init__(self, 
                 graph_weight: float = 0.3,
                 threshold_adjustment: float = 0.1):
        """
        Initialize graph-informed integration
        
        Args:
            graph_weight: Weight for graph similarity (0-1)
            threshold_adjustment: How much to adjust threshold based on graph
        """
        self.graph_weight = graph_weight
        self.vector_weight = 1.0 - graph_weight
        self.threshold_adjustment = threshold_adjustment
        
    def get_graph_connection_strength(self, 
                                    graph,
                                    node1: int, 
                                    node2: int) -> float:
        """
        Get connection strength between two nodes in graph
        
        Args:
            graph: PyTorch Geometric graph
            node1, node2: Node indices
            
        Returns:
            Connection strength (0-1)
        """
        if graph is None or not hasattr(graph, 'edge_index'):
            return 0.0
            
        edge_index = graph.edge_index
        
        # Check direct connection
        mask = ((edge_index[0] == node1) & (edge_index[1] == node2)) | \
               ((edge_index[0] == node2) & (edge_index[1] == node1))
        
        if mask.any():
            # Direct connection exists
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                # Use edge weight if available
                edge_idx = mask.nonzero()[0].item()
                return float(graph.edge_attr[edge_idx])
            else:
                # No edge weights, use fixed value for direct connection
                return 0.8
        
        # Check 2-hop connection (weaker)
        neighbors1 = edge_index[1][edge_index[0] == node1]
        neighbors2 = edge_index[1][edge_index[0] == node2]
        
        common_neighbors = set(neighbors1.tolist()) & set(neighbors2.tolist())
        if common_neighbors:
            return 0.4  # 2-hop connection
            
        return 0.0  # No connection
    
    def calculate_combined_similarity(self,
                                    vec1: np.ndarray,
                                    vec2: np.ndarray,
                                    graph_connection: float) -> float:
        """
        Calculate similarity combining vector and graph information
        
        Args:
            vec1, vec2: Episode vectors
            graph_connection: Strength of graph connection (0-1)
            
        Returns:
            Combined similarity score
        """
        # Vector similarity (cosine)
        vec_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        # Dynamic weighting based on graph connection
        # Strong graph connection increases its influence
        adj_graph_weight = self.graph_weight * (1 + 0.5 * graph_connection)
        adj_vector_weight = 1 - adj_graph_weight
        
        # Normalize weights
        total_weight = adj_graph_weight + adj_vector_weight
        adj_graph_weight /= total_weight
        adj_vector_weight /= total_weight
        
        # Combined score
        combined = adj_vector_weight * vec_sim + adj_graph_weight * graph_connection
        
        return combined
    
    def check_integration_with_graph(self,
                                   new_vector: np.ndarray,
                                   new_text: str,
                                   episodes: List[Dict],
                                   graph,
                                   new_node_idx: Optional[int] = None) -> Dict:
        """
        Check if new episode should integrate with existing ones using graph
        
        Args:
            new_vector: Vector of new episode
            new_text: Text of new episode
            episodes: List of existing episodes
            graph: Current knowledge graph from Layer3
            new_node_idx: Index of new node in graph (if already added)
            
        Returns:
            Integration decision with scores
        """
        if not episodes:
            return {
                "should_integrate": False,
                "target_index": -1,
                "reason": "no_existing_episodes"
            }
        
        best_candidate = {
            "index": -1,
            "vector_similarity": 0.0,
            "graph_connection": 0.0,
            "combined_score": 0.0,
            "content_overlap": 0.0
        }
        
        # If new node not in graph yet, we can only use vector similarity
        use_graph = (graph is not None and new_node_idx is not None)
        
        for i, episode in enumerate(episodes):
            # Vector similarity
            vec_sim = np.dot(new_vector, episode['vec']) / (
                np.linalg.norm(new_vector) * np.linalg.norm(episode['vec'])
            )
            
            # Graph connection
            if use_graph:
                graph_conn = self.get_graph_connection_strength(graph, i, new_node_idx)
            else:
                graph_conn = 0.0
            
            # Combined similarity
            combined = self.calculate_combined_similarity(
                new_vector, episode['vec'], graph_conn
            )
            
            # Content overlap (simple word-based)
            new_words = set(new_text.lower().split())
            episode_words = set(episode['text'].lower().split())
            content_overlap = len(new_words & episode_words) / len(new_words | episode_words) if new_words else 0
            
            # Update best candidate
            if combined > best_candidate["combined_score"]:
                best_candidate = {
                    "index": i,
                    "vector_similarity": vec_sim,
                    "graph_connection": graph_conn,
                    "combined_score": combined,
                    "content_overlap": content_overlap
                }
        
        # Adjust threshold based on graph connection
        base_threshold = 0.85
        base_content_threshold = 0.7
        
        if best_candidate["graph_connection"] > 0.5:
            # Strong graph connection lowers thresholds
            adjusted_threshold = base_threshold - self.threshold_adjustment
            adjusted_content_threshold = base_content_threshold - self.threshold_adjustment
            reason = "graph_boosted"
        else:
            adjusted_threshold = base_threshold
            adjusted_content_threshold = base_content_threshold
            reason = "standard"
        
        # Integration decision
        should_integrate = (
            best_candidate["combined_score"] >= adjusted_threshold and
            best_candidate["content_overlap"] >= adjusted_content_threshold
        )
        
        return {
            "should_integrate": should_integrate,
            "target_index": best_candidate["index"] if should_integrate else -1,
            "scores": best_candidate,
            "thresholds_used": {
                "similarity": adjusted_threshold,
                "content": adjusted_content_threshold
            },
            "reason": reason if should_integrate else "below_threshold"
        }


# Proposed modification to L2MemoryManager
def enhanced_check_episode_integration(self, vector: np.ndarray, text: str, c_value: float) -> Dict:
    """
    Enhanced version of _check_episode_integration using graph information
    
    This would replace the existing method in L2MemoryManager
    """
    # Get current graph from Layer3 if available
    graph = None
    if hasattr(self, 'l3_graph') and self.l3_graph and hasattr(self.l3_graph, 'previous_graph'):
        graph = self.l3_graph.previous_graph
    
    # Convert episodes to required format
    episodes = []
    for ep in self.episodes:
        episodes.append({
            'vec': ep.vec,
            'text': ep.text,
            'c': ep.c
        })
    
    # Use graph-informed integration
    integrator = GraphInformedIntegration()
    
    # For this example, assume new episode would be added as next node
    new_node_idx = len(self.episodes) if graph else None
    
    result = integrator.check_integration_with_graph(
        vector, text, episodes, graph, new_node_idx
    )
    
    # Convert to expected format
    return {
        "should_integrate": result["should_integrate"],
        "target_index": result["target_index"],
        "best_candidate": result["scores"],
        "reason": result["reason"]
    }


if __name__ == "__main__":
    print("=== Graph-Informed Integration Implementation ===\n")
    
    print("Key Features:")
    print("1. Combines vector similarity with graph structure")
    print("2. Dynamic weight adjustment based on connection strength")
    print("3. Threshold reduction for strongly connected nodes")
    print("4. Supports both direct and 2-hop connections")
    
    print("\nIntegration into Layer2:")
    print("- Replace _check_episode_integration with graph-aware version")
    print("- Access Layer3's graph for connection information")
    print("- Maintain backward compatibility when graph unavailable")
    
    print("\nExpected Benefits:")
    print("- Better integration of semantically related episodes")
    print("- Reduced false negatives (missing integrations)")
    print("- Knowledge graph structure informs memory organization")