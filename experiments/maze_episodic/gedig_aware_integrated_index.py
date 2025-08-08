#!/usr/bin/env python3
"""
geDIG-aware Integrated Vector-Graph Index
Combines similarity evaluation with geDIG values for edge selection
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import time
from collections import defaultdict


class GeDIGAwareIntegratedIndex:
    """
    Integrated index that uses both similarity and geDIG values for edge selection
    """
    
    def __init__(self, dimension: int, config: Optional[Dict] = None):
        self.dimension = dimension
        self.config = config or {}
        
        # Storage
        self.normalized_vectors = []
        self.vector_norms = []
        self.metadata = []
        self.graph = nx.Graph()
        
        # geDIG parameters
        self.gedig_threshold = self.config.get('gedig_threshold', 0.5)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        self.max_edges_per_node = self.config.get('max_edges_per_node', 10)
        self.gedig_weight = self.config.get('gedig_weight', 0.5)  # Balance between similarity and geDIG
        
        # Spatial index for position-based queries
        self.spatial_index = defaultdict(list)
        
        # Performance tracking
        self.stats = {
            'gedig_calculations': 0,
            'edge_additions': 0,
            'edge_rejections': 0
        }
    
    def _calculate_gedig_value(self, idx1: int, idx2: int, 
                              similarity: float) -> Tuple[float, Dict]:
        """
        Calculate geDIG value between two episodes
        
        geDIG = GED - IG
        - GED (Graph Edit Distance): How different the graph structures are
        - IG (Information Gain): How much new information is gained
        
        Returns:
            (gedig_value, details)
        """
        self.stats['gedig_calculations'] += 1
        
        # Get metadata
        meta1 = self.metadata[idx1]
        meta2 = self.metadata[idx2]
        
        # 1. Calculate Graph Edit Distance (GED)
        # For maze navigation, we consider:
        # - Spatial distance (how far apart in the maze)
        # - Temporal distance (when encountered)
        # - Action difference (what actions were taken)
        
        # Spatial distance
        pos1 = meta1.get('pos', (0, 0))
        pos2 = meta2.get('pos', (0, 0))
        spatial_dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        normalized_spatial = spatial_dist / (np.sqrt(2) * 50)  # Normalize by max possible distance
        
        # Temporal distance (based on episode indices)
        temporal_dist = abs(idx1 - idx2) / max(len(self.metadata), 1)
        
        # Action similarity (if actions are stored)
        action1 = meta1.get('action', None)
        action2 = meta2.get('action', None)
        action_similarity = 1.0 if action1 == action2 else 0.0
        
        # GED combines these factors
        ged = (normalized_spatial * 0.4 + 
               temporal_dist * 0.3 + 
               (1 - action_similarity) * 0.3)
        
        # 2. Calculate Information Gain (IG)
        # High similarity = low information gain
        # Episodes from different contexts = high information gain
        
        # Context difference
        c_value1 = meta1.get('c_value', 0.5)
        c_value2 = meta2.get('c_value', 0.5)
        confidence_diff = abs(c_value1 - c_value2)
        
        # Information gain is inversely related to similarity
        # but positively related to context difference
        ig = (1 - similarity) * 0.7 + confidence_diff * 0.3
        
        # 3. Calculate geDIG value
        gedig = ged - ig
        
        # 4. Determine if this edge should be created
        # Lower geDIG is better (less distance, more information gain)
        # But we also want some similarity
        
        details = {
            'ged': ged,
            'ig': ig,
            'gedig': gedig,
            'spatial_dist': spatial_dist,
            'temporal_dist': temporal_dist,
            'action_similarity': action_similarity,
            'confidence_diff': confidence_diff
        }
        
        return gedig, details
    
    def _evaluate_edge_candidate(self, idx1: int, idx2: int, 
                                similarity: float) -> Tuple[bool, float, Dict]:
        """
        Evaluate whether an edge should be created between two episodes
        
        Returns:
            (should_create, combined_score, details)
        """
        # Calculate geDIG value
        gedig_value, gedig_details = self._calculate_gedig_value(idx1, idx2, similarity)
        
        # Combine similarity and geDIG scores
        # For geDIG: lower is better, so we use (1 - normalized_gedig)
        # Normalize geDIG to [0, 1] range
        normalized_gedig = np.clip(gedig_value, -1, 1)  # geDIG can be negative (good)
        gedig_score = 1.0 - (normalized_gedig + 1) / 2  # Convert to [0, 1] where 1 is good
        
        # Combined score
        combined_score = (similarity * (1 - self.gedig_weight) + 
                         gedig_score * self.gedig_weight)
        
        # Decision criteria
        should_create = (
            similarity >= self.similarity_threshold and
            gedig_value < self.gedig_threshold and
            combined_score > 0.6  # Combined threshold
        )
        
        details = {
            **gedig_details,
            'similarity': similarity,
            'gedig_score': gedig_score,
            'combined_score': combined_score,
            'decision': should_create
        }
        
        return should_create, combined_score, details
    
    def add_episode(self, episode: Dict) -> int:
        """
        Add episode with geDIG-aware edge creation
        """
        # Extract and normalize vector
        vec = np.array(episode['vec'], dtype=np.float32)
        norm = np.linalg.norm(vec)
        
        if norm == 0:
            normalized_vec = vec
        else:
            normalized_vec = vec / norm
        
        # Add to storage
        idx = len(self.normalized_vectors)
        self.normalized_vectors.append(normalized_vec)
        self.vector_norms.append(norm)
        self.metadata.append(episode)
        
        # Add node to graph
        self.graph.add_node(idx, **episode)
        
        # Update spatial index
        if 'pos' in episode:
            pos_key = f"{episode['pos'][0]},{episode['pos'][1]}"
            self.spatial_index[pos_key].append(idx)
        
        # Add edges with geDIG evaluation
        if len(self.normalized_vectors) > 1:
            self._add_gedig_aware_edges(idx, normalized_vec)
        
        return idx
    
    def _add_gedig_aware_edges(self, idx: int, normalized_vec: np.ndarray):
        """
        Add edges based on both similarity and geDIG values
        """
        # Calculate similarities with all existing episodes
        existing_vectors = np.array(self.normalized_vectors[:-1])
        similarities = np.dot(existing_vectors, normalized_vec)
        
        # Find candidates above similarity threshold
        candidates = []
        for i, sim in enumerate(similarities):
            if sim >= self.similarity_threshold * 0.8:  # Slightly lower threshold for candidates
                should_create, combined_score, details = self._evaluate_edge_candidate(
                    idx, i, sim
                )
                
                if should_create:
                    candidates.append((i, combined_score, details))
        
        # Sort by combined score and take top k
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Add edges for top candidates
        edges_added = 0
        for sim_idx, score, details in candidates[:self.max_edges_per_node]:
            self.graph.add_edge(
                idx, sim_idx,
                weight=score,
                similarity=details['similarity'],
                gedig=details['gedig'],
                ged=details['ged'],
                ig=details['ig']
            )
            edges_added += 1
            self.stats['edge_additions'] += 1
        
        # Track rejections
        self.stats['edge_rejections'] += len(candidates) - edges_added
    
    def search(self, query_vector: np.ndarray, k: int = 10, 
               mode: str = 'vector', **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search with various modes
        """
        query_vector = np.array(query_vector, dtype=np.float32)
        
        # Normalize query
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            normalized_query = query_vector
        else:
            normalized_query = query_vector / query_norm
        
        if mode == 'vector':
            return self._vector_search(normalized_query, k)
        elif mode == 'gedig':
            return self._gedig_guided_search(normalized_query, k, **kwargs)
        elif mode == 'hybrid':
            return self._hybrid_search(normalized_query, k, **kwargs)
        else:
            raise ValueError(f"Unknown search mode: {mode}")
    
    def _vector_search(self, normalized_query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pure vector similarity search
        """
        if len(self.normalized_vectors) == 0:
            return np.array([]), np.array([])
        
        vectors_array = np.array(self.normalized_vectors)
        similarities = np.dot(vectors_array, normalized_query)
        
        # Get top-k
        if len(similarities) <= k:
            indices = np.arange(len(similarities))
            scores = similarities
        else:
            top_indices = np.argpartition(similarities, -k)[-k:]
            indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
            scores = similarities[indices]
        
        return indices, scores
    
    def _gedig_guided_search(self, normalized_query: np.ndarray, k: int, 
                            start_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search guided by geDIG values through the graph
        """
        if len(self.normalized_vectors) == 0:
            return np.array([]), np.array([])
        
        # Start from most similar node if not specified
        if start_idx is None:
            initial_indices, initial_scores = self._vector_search(normalized_query, 1)
            if len(initial_indices) == 0:
                return np.array([]), np.array([])
            start_idx = initial_indices[0]
        
        # Traverse graph following low geDIG edges
        visited = set()
        candidates = []
        
        # BFS with geDIG-weighted priority
        queue = [(start_idx, 1.0)]
        
        while queue and len(candidates) < k * 3:
            current_idx, current_score = queue.pop(0)
            
            if current_idx in visited:
                continue
            
            visited.add(current_idx)
            
            # Calculate similarity to query
            vec = self.normalized_vectors[current_idx]
            similarity = np.dot(vec, normalized_query)
            combined_score = current_score * 0.3 + similarity * 0.7
            candidates.append((current_idx, combined_score))
            
            # Explore neighbors with good geDIG values
            if current_idx in self.graph:
                for neighbor in self.graph.neighbors(current_idx):
                    if neighbor not in visited:
                        edge_data = self.graph[current_idx][neighbor]
                        gedig_value = edge_data.get('gedig', 1.0)
                        
                        # Prioritize low geDIG edges
                        neighbor_score = current_score * (1.0 - gedig_value)
                        queue.append((neighbor, neighbor_score))
        
        # Sort by score and return top k
        candidates.sort(key=lambda x: x[1], reverse=True)
        indices = np.array([c[0] for c in candidates[:k]])
        scores = np.array([c[1] for c in candidates[:k]])
        
        return indices, scores
    
    def _hybrid_search(self, normalized_query: np.ndarray, k: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hybrid search combining vector similarity and geDIG-guided graph traversal
        """
        # Get initial candidates from vector search
        vec_indices, vec_scores = self._vector_search(normalized_query, k // 2)
        
        # Expand through geDIG-guided search
        gedig_candidates = []
        for idx in vec_indices[:3]:  # Start from top 3 similar
            g_indices, g_scores = self._gedig_guided_search(
                normalized_query, k // 3, start_idx=int(idx)
            )
            for i, s in zip(g_indices, g_scores):
                gedig_candidates.append((i, s))
        
        # Combine and deduplicate
        all_candidates = {}
        for idx, score in zip(vec_indices, vec_scores):
            all_candidates[int(idx)] = score
        
        for idx, score in gedig_candidates:
            if idx in all_candidates:
                all_candidates[idx] = max(all_candidates[idx], score)
            else:
                all_candidates[idx] = score
        
        # Sort and return top k
        sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)
        indices = np.array([c[0] for c in sorted_candidates[:k]])
        scores = np.array([c[1] for c in sorted_candidates[:k]])
        
        return indices, scores
    
    def get_statistics(self) -> Dict:
        """Get index statistics"""
        return {
            'episodes': len(self.metadata),
            'edges': self.graph.number_of_edges(),
            'avg_degree': self.graph.number_of_edges() * 2 / max(self.graph.number_of_nodes(), 1),
            'gedig_calculations': self.stats['gedig_calculations'],
            'edge_additions': self.stats['edge_additions'],
            'edge_rejections': self.stats['edge_rejections'],
            'edge_acceptance_rate': self.stats['edge_additions'] / max(
                self.stats['edge_additions'] + self.stats['edge_rejections'], 1
            )
        }