#!/usr/bin/env python3
"""
GNN-guided episode integration
Use graph edge weights to inform episode integration decisions
"""

import os
import sys
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from insightspike.core.layers.scalable_graph_builder import ScalableGraphBuilder
from torch_geometric.data import Data


class GNNGuidedIntegration:
    """
    Enhanced episode integration using GNN edge weights
    """
    
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.graph_builder = ScalableGraphBuilder()
        self.current_graph = None
        self.edge_weights = None
        
    def build_weighted_graph(self, documents: List[Dict]) -> Data:
        """
        Build graph with edge weights based on similarity
        """
        # Build base graph
        graph = self.graph_builder.build_graph(documents)
        
        # Calculate edge weights
        if graph.edge_index.size(1) > 0:
            edge_weights = []
            edge_index = graph.edge_index.numpy()
            
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i], edge_index[1, i]
                
                # Calculate similarity between connected nodes
                src_emb = graph.x[src].numpy()
                dst_emb = graph.x[dst].numpy()
                
                similarity = np.dot(src_emb, dst_emb) / (
                    np.linalg.norm(src_emb) * np.linalg.norm(dst_emb)
                )
                edge_weights.append(similarity)
            
            # Add edge weights to graph
            graph.edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
        else:
            graph.edge_attr = torch.empty(0, dtype=torch.float32)
            
        self.current_graph = graph
        self.edge_weights = graph.edge_attr
        
        return graph
    
    def get_graph_similarity(self, node_idx1: int, node_idx2: int) -> float:
        """
        Get similarity between two nodes based on graph structure
        """
        if self.current_graph is None:
            return 0.0
            
        # Direct connection weight
        edge_index = self.current_graph.edge_index
        edge_mask = ((edge_index[0] == node_idx1) & (edge_index[1] == node_idx2)) | \
                   ((edge_index[0] == node_idx2) & (edge_index[1] == node_idx1))
        
        if edge_mask.any():
            # Direct connection exists
            edge_idx = edge_mask.nonzero()[0].item()
            return float(self.edge_weights[edge_idx])
        
        # No direct connection - use path-based similarity
        return self._path_similarity(node_idx1, node_idx2)
    
    def _path_similarity(self, node1: int, node2: int, max_depth: int = 2) -> float:
        """
        Calculate similarity based on shortest path in graph
        """
        # Simple BFS to find shortest path
        from collections import deque
        
        if node1 == node2:
            return 1.0
            
        visited = {node1}
        queue = deque([(node1, 0, 1.0)])  # (node, depth, accumulated_weight)
        
        edge_index = self.current_graph.edge_index.numpy()
        
        while queue:
            current, depth, weight = queue.popleft()
            
            if depth >= max_depth:
                continue
                
            # Find neighbors
            neighbors_mask = edge_index[0] == current
            neighbors = edge_index[1][neighbors_mask]
            neighbor_weights = self.edge_weights[neighbors_mask].numpy()
            
            for neighbor, edge_weight in zip(neighbors, neighbor_weights):
                if neighbor == node2:
                    # Found path to target
                    return weight * edge_weight * (0.8 ** depth)  # Decay by depth
                    
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1, weight * edge_weight))
        
        return 0.0  # No path found
    
    def enhanced_integration_check(self, 
                                 new_vector: np.ndarray,
                                 new_text: str,
                                 existing_episodes: List[Dict],
                                 threshold_adjustment: float = 0.1) -> Dict:
        """
        Check integration with GNN guidance
        """
        if not existing_episodes:
            return {"should_integrate": False, "target_index": -1}
        
        # Build current graph
        documents = [{"text": ep["text"], "embedding": ep["vector"], "id": i} 
                    for i, ep in enumerate(existing_episodes)]
        documents.append({"text": new_text, "embedding": new_vector, "id": len(documents)})
        
        graph = self.build_weighted_graph(documents)
        new_node_idx = len(documents) - 1
        
        best_candidate = {
            "index": -1,
            "vector_similarity": 0.0,
            "graph_similarity": 0.0,
            "combined_score": 0.0
        }
        
        # Check each existing episode
        for i in range(len(existing_episodes)):
            # Traditional vector similarity
            vec_sim = np.dot(new_vector, existing_episodes[i]["vector"]) / (
                np.linalg.norm(new_vector) * np.linalg.norm(existing_episodes[i]["vector"])
            )
            
            # Graph-based similarity
            graph_sim = self.get_graph_similarity(i, new_node_idx)
            
            # Combined score (weighted average)
            combined = 0.6 * vec_sim + 0.4 * graph_sim
            
            if combined > best_candidate["combined_score"]:
                best_candidate = {
                    "index": i,
                    "vector_similarity": vec_sim,
                    "graph_similarity": graph_sim,
                    "combined_score": combined
                }
        
        # Adjust threshold based on graph connectivity
        base_threshold = 0.85
        if best_candidate["graph_similarity"] > 0.5:
            # Strong graph connection lowers the threshold
            adjusted_threshold = base_threshold - threshold_adjustment
        else:
            adjusted_threshold = base_threshold
        
        should_integrate = best_candidate["combined_score"] >= adjusted_threshold
        
        return {
            "should_integrate": should_integrate,
            "target_index": best_candidate["index"] if should_integrate else -1,
            "scores": best_candidate,
            "threshold_used": adjusted_threshold
        }


def test_gnn_integration():
    """Test GNN-guided integration"""
    print("=== Testing GNN-Guided Episode Integration ===\n")
    
    # Create test episodes
    episodes = []
    
    # Cluster 1: AI/ML topics
    for i in range(3):
        vec = np.random.randn(384)
        vec[0:100] = np.random.randn(100) * 2  # Strong signal in first quarter
        vec = vec / np.linalg.norm(vec)
        episodes.append({
            "text": f"Machine learning research paper {i}",
            "vector": vec.astype(np.float32)
        })
    
    # Cluster 2: Biology topics
    for i in range(3):
        vec = np.random.randn(384)
        vec[200:300] = np.random.randn(100) * 2  # Strong signal in third quarter
        vec = vec / np.linalg.norm(vec)
        episodes.append({
            "text": f"Biology research findings {i}",
            "vector": vec.astype(np.float32)
        })
    
    # Initialize GNN integration
    integrator = GNNGuidedIntegration()
    
    # Test cases
    test_cases = [
        # Similar to ML cluster
        {
            "text": "Deep learning advances in neural networks",
            "vector": lambda: np.concatenate([np.random.randn(100)*2, np.random.randn(284)]) / 5
        },
        # Similar to biology cluster
        {
            "text": "Genetic research breakthrough",
            "vector": lambda: np.concatenate([np.random.randn(200), np.random.randn(100)*2, np.random.randn(84)]) / 5
        },
        # Mixed topic
        {
            "text": "AI applications in biological research",
            "vector": lambda: np.concatenate([np.random.randn(100), np.zeros(100), np.random.randn(100), np.random.randn(84)]) / 3
        }
    ]
    
    print("Current episodes:")
    for i, ep in enumerate(episodes):
        print(f"  {i}: {ep['text']}")
    
    print("\n" + "="*50 + "\n")
    
    for test in test_cases:
        vec = test["vector"]()
        vec = vec / np.linalg.norm(vec)
        vec = vec.astype(np.float32)
        
        result = integrator.enhanced_integration_check(vec, test["text"], episodes)
        
        print(f"Test: {test['text']}")
        print(f"  Should integrate: {result['should_integrate']}")
        if result['should_integrate']:
            print(f"  Target episode: {result['target_index']} - {episodes[result['target_index']]['text']}")
        print(f"  Scores:")
        print(f"    Vector similarity: {result['scores']['vector_similarity']:.3f}")
        print(f"    Graph similarity: {result['scores']['graph_similarity']:.3f}")
        print(f"    Combined score: {result['scores']['combined_score']:.3f}")
        print(f"  Threshold used: {result['threshold_used']:.3f}")
        print()


if __name__ == "__main__":
    test_gnn_integration()