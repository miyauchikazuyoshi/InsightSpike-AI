"""Dynamic knowledge graph management for geDIG-RAG v3."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path
import json

import networkx as nx
import numpy as np

from .gedig_evaluator import GeDIGEvaluator, GraphUpdate, UpdateType


@dataclass
class KnowledgeNode:
    """Represents a knowledge node in the graph."""
    node_id: str
    text: str
    embedding: np.ndarray
    node_type: str  # fact, question, summary, insight, definition
    confidence: float
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.last_accessed == 0.0:
            self.last_accessed = self.timestamp
        if self.metadata is None:
            self.metadata = {}
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'node_id': self.node_id,
            'text': self.text,
            'embedding': self.embedding.tolist(),
            'node_type': self.node_type,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KnowledgeNode:
        """Create from dictionary."""
        data_copy = data.copy()
        data_copy['embedding'] = np.array(data_copy['embedding'])
        return cls(**data_copy)


@dataclass
class KnowledgeEdge:
    """Represents an edge between knowledge nodes."""
    source: str
    target: str
    relation: str  # semantic, temporal, causal, entailment, contradiction
    weight: float
    
    # Edge features for geDIG analysis
    semantic_similarity: float
    lexical_overlap: float = 0.0
    temporal_distance: float = 0.0
    entailment_score: float = 0.0
    contradiction_score: float = 0.0
    causal_likelihood: float = 0.0
    
    # Access statistics
    access_count: int = 0
    last_used: float = 0.0
    
    def __post_init__(self):
        if self.last_used == 0.0:
            self.last_used = time.time()
    
    def update_usage(self):
        """Update usage statistics."""
        self.access_count += 1
        self.last_used = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'source': self.source,
            'target': self.target,
            'relation': self.relation,
            'weight': self.weight,
            'semantic_similarity': self.semantic_similarity,
            'lexical_overlap': self.lexical_overlap,
            'temporal_distance': self.temporal_distance,
            'entailment_score': self.entailment_score,
            'contradiction_score': self.contradiction_score,
            'causal_likelihood': self.causal_likelihood,
            'access_count': self.access_count,
            'last_used': self.last_used
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KnowledgeEdge:
        """Create from dictionary."""
        return cls(**data)


class KnowledgeGraph:
    """Dynamic knowledge graph with geDIG-based updates."""
    
    def __init__(self, embedding_dim: int = 384):
        """Initialize knowledge graph.
        
        Args:
            embedding_dim: Dimension of node embeddings
        """
        self.graph = nx.Graph()
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[Tuple[str, str], KnowledgeEdge] = {}
        self.embedding_dim = embedding_dim
        
        # Statistics tracking
        self.stats = {
            'nodes_added': 0,
            'nodes_pruned': 0,
            'nodes_merged': 0,
            'edges_added': 0,
            'edges_removed': 0,
            'total_accesses': 0,
            'creation_time': time.time()
        }
        
        # Performance tracking
        self.operation_times: Dict[str, List[float]] = {
            'add_node': [],
            'remove_node': [],
            'add_edge': [],
            'search': []
        }
    
    def add_node(self,
                text: str,
                embedding: Optional[np.ndarray] = None,
                node_type: str = "fact",
                confidence: float = 1.0,
                metadata: Optional[Dict] = None,
                node_id: Optional[str] = None) -> str:
        """Add a new knowledge node.
        
        Args:
            text: Text content of the node
            embedding: Node embedding vector
            node_type: Type of knowledge node
            confidence: Confidence score
            metadata: Additional metadata
            node_id: Optional custom node ID
            
        Returns:
            ID of the created node
        """
        start_time = time.time()
        
        if node_id is None:
            node_id = str(uuid.uuid4())
        
        if embedding is None:
            # Create dummy embedding (should be replaced with actual embedder)
            embedding = np.random.normal(0, 1, self.embedding_dim)
        
        # Create knowledge node
        node = KnowledgeNode(
            node_id=node_id,
            text=text,
            embedding=embedding,
            node_type=node_type,
            confidence=confidence,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        # Add to graph structures
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **node.to_dict())
        
        # Update statistics
        self.stats['nodes_added'] += 1
        self.operation_times['add_node'].append(time.time() - start_time)
        
        return node_id
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a knowledge node and its edges.
        
        Args:
            node_id: ID of node to remove
            
        Returns:
            True if node was removed, False if not found
        """
        start_time = time.time()
        
        if node_id not in self.nodes:
            return False
        
        # Remove associated edges
        edges_to_remove = []
        for edge_key, edge in self.edges.items():
            if edge.source == node_id or edge.target == node_id:
                edges_to_remove.append(edge_key)
        
        for edge_key in edges_to_remove:
            del self.edges[edge_key]
            self.stats['edges_removed'] += 1
        
        # Remove node
        del self.nodes[node_id]
        self.graph.remove_node(node_id)
        
        # Update statistics
        self.stats['nodes_pruned'] += 1
        self.operation_times['remove_node'].append(time.time() - start_time)
        
        return True
    
    def add_edge(self,
                source: str,
                target: str,
                relation: str = "semantic",
                weight: float = 1.0,
                semantic_similarity: float = 0.0,
                **edge_kwargs) -> bool:
        """Add an edge between two knowledge nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            relation: Type of relation
            weight: Edge weight
            semantic_similarity: Semantic similarity score
            **edge_kwargs: Additional edge attributes
            
        Returns:
            True if edge was added successfully
        """
        if source not in self.nodes or target not in self.nodes:
            return False
        
        # Create edge key (undirected)
        edge_key = (min(source, target), max(source, target))
        
        # Create knowledge edge
        edge = KnowledgeEdge(
            source=source,
            target=target,
            relation=relation,
            weight=weight,
            semantic_similarity=semantic_similarity,
            **edge_kwargs
        )
        
        # Add to graph structures
        self.edges[edge_key] = edge
        self.graph.add_edge(source, target, **edge.to_dict())
        
        # Update statistics
        self.stats['edges_added'] += 1
        
        return True
    
    def find_similar_nodes(self,
                          query_embedding: np.ndarray,
                          k: int = 5,
                          min_similarity: float = 0.0,
                          node_types: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """Find nodes similar to query embedding.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of similar nodes to return
            min_similarity: Minimum similarity threshold
            node_types: Filter by node types
            
        Returns:
            List of (node_id, similarity) tuples
        """
        start_time = time.time()
        
        similarities = []
        
        for node_id, node in self.nodes.items():
            # Filter by node type if specified
            if node_types and node.node_type not in node_types:
                continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, node.embedding)
            
            if similarity >= min_similarity:
                similarities.append((node_id, similarity))
                # Update access statistics
                node.update_access()
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        result = similarities[:k]
        
        # Update statistics
        self.stats['total_accesses'] += len(result)
        self.operation_times['search'].append(time.time() - start_time)
        
        return result
    
    def get_node_neighbors(self, node_id: str, max_distance: int = 1) -> List[Tuple[str, int]]:
        """Get neighbors of a node within max distance.
        
        Args:
            node_id: Center node ID
            max_distance: Maximum distance to search
            
        Returns:
            List of (neighbor_id, distance) tuples
        """
        if node_id not in self.nodes:
            return []
        
        neighbors = []
        
        # BFS to find neighbors within max distance
        visited = set()
        queue = [(node_id, 0)]
        
        while queue:
            current_node, distance = queue.pop(0)
            
            if current_node in visited or distance > max_distance:
                continue
            
            visited.add(current_node)
            
            if distance > 0:  # Don't include the center node itself
                neighbors.append((current_node, distance))
            
            # Add neighbors to queue
            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))
        
        return neighbors
    
    def update_node_embedding(self, node_id: str, new_embedding: np.ndarray) -> bool:
        """Update a node's embedding.
        
        Args:
            node_id: Node to update
            new_embedding: New embedding vector
            
        Returns:
            True if updated successfully
        """
        if node_id not in self.nodes:
            return False
        
        self.nodes[node_id].embedding = new_embedding
        # Update graph node attributes
        self.graph.nodes[node_id]['embedding'] = new_embedding.tolist()
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics.
        
        Returns:
            Dictionary of graph statistics
        """
        current_time = time.time()
        
        # Basic graph metrics
        n_nodes = len(self.nodes)
        n_edges = len(self.edges)
        
        # Performance metrics
        avg_times = {}
        for operation, times in self.operation_times.items():
            avg_times[f'avg_{operation}_time'] = np.mean(times) if times else 0.0
        
        # Node type distribution
        node_type_counts = {}
        for node in self.nodes.values():
            node_type_counts[node.node_type] = node_type_counts.get(node.node_type, 0) + 1
        
        # Access patterns
        access_counts = [node.access_count for node in self.nodes.values()]
        
        return {
            'current_nodes': n_nodes,
            'current_edges': n_edges,
            'nodes_added': self.stats['nodes_added'],
            'nodes_pruned': self.stats['nodes_pruned'],
            'nodes_merged': self.stats['nodes_merged'],
            'edges_added': self.stats['edges_added'],
            'edges_removed': self.stats['edges_removed'],
            'total_accesses': self.stats['total_accesses'],
            'graph_density': n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0.0,
            'avg_node_degree': np.mean([d for n, d in self.graph.degree()]) if n_nodes > 0 else 0.0,
            'node_type_distribution': node_type_counts,
            'access_statistics': {
                'mean_access_count': np.mean(access_counts) if access_counts else 0.0,
                'max_access_count': np.max(access_counts) if access_counts else 0.0,
                'total_accesses': sum(access_counts)
            },
            'performance_metrics': avg_times,
            'uptime_seconds': current_time - self.stats['creation_time']
        }
    
    def save_to_file(self, file_path: Path) -> None:
        """Save graph to JSON file.
        
        Args:
            file_path: Path to save file
        """
        # Prepare data for serialization
        data = {
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            'edges': {f"{k[0]}_{k[1]}": edge.to_dict() for k, edge in self.edges.items()},
            'stats': self.stats,
            'embedding_dim': self.embedding_dim
        }
        
        # Save to file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, file_path: Path) -> None:
        """Load graph from JSON file.
        
        Args:
            file_path: Path to load from
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Clear current state
        self.graph.clear()
        self.nodes.clear()
        self.edges.clear()
        
        # Load basic parameters
        self.embedding_dim = data.get('embedding_dim', 384)
        self.stats = data.get('stats', {})
        
        # Load nodes
        for node_id, node_data in data['nodes'].items():
            node = KnowledgeNode.from_dict(node_data)
            self.nodes[node_id] = node
            self.graph.add_node(node_id, **node.to_dict())
        
        # Load edges
        for edge_key_str, edge_data in data['edges'].items():
            parts = edge_key_str.split('_', 1)
            edge_key = (parts[0], parts[1])
            edge = KnowledgeEdge.from_dict(edge_data)
            self.edges[edge_key] = edge
            self.graph.add_edge(edge.source, edge.target, **edge.to_dict())
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(vec1, vec2)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norms == 0:
            return 0.0
        
        return dot_product / norms
    
    def get_connected_components(self) -> List[Set[str]]:
        """Get connected components in the graph.
        
        Returns:
            List of sets, each containing node IDs in a connected component
        """
        return list(nx.connected_components(self.graph))
    
    def get_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Get shortest path between two nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            List of node IDs forming shortest path, or None if no path exists
        """
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None