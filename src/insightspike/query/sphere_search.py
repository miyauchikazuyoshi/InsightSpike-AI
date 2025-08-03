"""
Query-centric sphere search implementation for wake mode processing.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import faiss
from dataclasses import dataclass


@dataclass
class NeighborNode:
    """A node found within the search sphere."""
    node_id: str
    distance: float
    relative_position: np.ndarray
    vector: np.ndarray


class SphereSearch:
    """
    Query-centric sphere search for finding relevant nodes.
    The query becomes the origin of a local coordinate system.
    """
    
    def __init__(self, node_vectors: Dict[str, np.ndarray]):
        """
        Initialize sphere search with node vectors.
        
        Args:
            node_vectors: Dictionary mapping node_id to vector representation
        """
        self.node_vectors = node_vectors
        self.node_ids = list(node_vectors.keys())
        self.vectors = np.array(list(node_vectors.values()))
        
        # Initialize FAISS index for efficient search
        self.dimension = self.vectors.shape[1]
        self.index = None
        self._build_index()
    
    def _build_index(self):
        """Build FAISS index for efficient similarity search."""
        if len(self.vectors) > 0:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(self.vectors.astype(np.float32))
    
    def search_sphere(
        self, 
        query_vec: np.ndarray, 
        radius: float,
        max_neighbors: Optional[int] = None
    ) -> List[NeighborNode]:
        """
        Find all nodes within radius from query point.
        
        Args:
            query_vec: Query vector (becomes origin of local coordinates)
            radius: Search radius
            max_neighbors: Maximum number of neighbors to return
            
        Returns:
            List of NeighborNode objects sorted by distance
        """
        if self.index is None or len(self.vectors) == 0:
            return []
        
        # Search for candidates (use 3x radius for safety)
        k = min(len(self.vectors), max_neighbors or len(self.vectors))
        
        # FAISS returns squared distances
        distances_sq, indices = self.index.search(
            query_vec.reshape(1, -1).astype(np.float32), 
            k=k
        )
        
        # Filter by actual radius and create NeighborNode objects
        neighbors = []
        for idx, dist_sq in zip(indices[0], distances_sq[0]):
            if idx == -1:  # FAISS returns -1 for padding
                continue
                
            distance = np.sqrt(dist_sq)
            if distance < radius:
                node_vec = self.vectors[idx]
                relative_pos = node_vec - query_vec
                
                neighbors.append(NeighborNode(
                    node_id=self.node_ids[idx],
                    distance=distance,
                    relative_position=relative_pos,
                    vector=node_vec
                ))
        
        # Sort by distance
        neighbors.sort(key=lambda x: x.distance)
        
        # Apply max_neighbors limit if specified
        if max_neighbors:
            neighbors = neighbors[:max_neighbors]
        
        return neighbors
    
    def search_donut(
        self,
        query_vec: np.ndarray,
        inner_radius: float,
        outer_radius: float,
        max_neighbors: Optional[int] = None
    ) -> List[NeighborNode]:
        """
        Donut search: Find nodes between inner and outer radius.
        Filters out both too-close (known) and too-far (irrelevant) nodes.
        
        Args:
            query_vec: Query vector
            inner_radius: Inner radius (exclude closer nodes)
            outer_radius: Outer radius (exclude farther nodes)
            max_neighbors: Maximum number of neighbors
            
        Returns:
            List of NeighborNode objects in the donut region
        """
        # First get all nodes within outer radius
        candidates = self.search_sphere(query_vec, outer_radius, max_neighbors=None)
        
        # Filter by inner radius
        donut_neighbors = [
            n for n in candidates 
            if n.distance > inner_radius
        ]
        
        # Apply limit if specified
        if max_neighbors:
            donut_neighbors = donut_neighbors[:max_neighbors]
        
        return donut_neighbors
    
    def get_statistics(self, neighbors: List[NeighborNode]) -> Dict:
        """
        Get statistics about the neighbor distribution.
        
        Args:
            neighbors: List of neighbor nodes
            
        Returns:
            Dictionary with statistics
        """
        if not neighbors:
            return {
                'count': 0,
                'min_distance': None,
                'max_distance': None,
                'mean_distance': None,
                'std_distance': None
            }
        
        distances = [n.distance for n in neighbors]
        
        return {
            'count': len(neighbors),
            'min_distance': min(distances),
            'max_distance': max(distances),
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances)
        }


class SimpleSphereSearch:
    """
    Simple sphere search without FAISS dependency.
    For small-scale experiments or when FAISS is not available.
    """
    
    def __init__(self, node_vectors: Dict[str, np.ndarray]):
        """Initialize with node vectors."""
        self.node_vectors = node_vectors
    
    def search_sphere(
        self,
        query_vec: np.ndarray,
        radius: float,
        max_neighbors: Optional[int] = None
    ) -> List[NeighborNode]:
        """
        Brute-force sphere search.
        
        Args:
            query_vec: Query vector
            radius: Search radius
            max_neighbors: Maximum neighbors to return
            
        Returns:
            List of neighbors within radius
        """
        neighbors = []
        
        for node_id, node_vec in self.node_vectors.items():
            # Calculate query-relative position
            relative_pos = node_vec - query_vec
            distance = np.linalg.norm(relative_pos)
            
            if distance < radius:
                neighbors.append(NeighborNode(
                    node_id=node_id,
                    distance=distance,
                    relative_position=relative_pos,
                    vector=node_vec
                ))
        
        # Sort by distance
        neighbors.sort(key=lambda x: x.distance)
        
        # Apply limit
        if max_neighbors:
            neighbors = neighbors[:max_neighbors]
        
        return neighbors
    
    def search_donut(
        self,
        query_vec: np.ndarray,
        inner_radius: float,
        outer_radius: float,
        max_neighbors: Optional[int] = None
    ) -> List[NeighborNode]:
        """Donut search using simple implementation."""
        neighbors = []
        
        for node_id, node_vec in self.node_vectors.items():
            relative_pos = node_vec - query_vec
            distance = np.linalg.norm(relative_pos)
            
            if inner_radius < distance < outer_radius:
                neighbors.append(NeighborNode(
                    node_id=node_id,
                    distance=distance,
                    relative_position=relative_pos,
                    vector=node_vec
                ))
        
        neighbors.sort(key=lambda x: x.distance)
        
        if max_neighbors:
            neighbors = neighbors[:max_neighbors]
        
        return neighbors