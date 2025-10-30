"""
Query Type-specific Processing
==============================

Implements different processing strategies for different query types:
- Language queries: Use high-dimensional semantic vectors
- Spatial/Maze queries: Use low-dimensional position vectors
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np

from .gedig_wake_mode import WakeModeGeDIG, WakeModeResult

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries with different processing needs."""
    LANGUAGE = "language"      # Natural language, high-dim vectors
    SPATIAL = "spatial"        # Maze/navigation, low-dim vectors
    HYBRID = "hybrid"          # Mixed queries
    UNKNOWN = "unknown"        # Default fallback


@dataclass
class QueryContext:
    """Enhanced query context with type information."""
    query_text: Optional[str] = None
    query_type: QueryType = QueryType.UNKNOWN
    position: Optional[Tuple[float, float]] = None
    direction: Optional[int] = None
    language_embedding: Optional[np.ndarray] = None
    spatial_bounds: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class QueryTypeProcessor:
    """
    Processes queries differently based on their type.
    
    Key features:
    - Automatic query type detection
    - Type-specific vector representations
    - Optimized processing strategies
    """
    
    def __init__(self,
                 language_dim: int = 384,      # Standard embedding dimension
                 spatial_dim: int = 5,         # Compact spatial dimension
                 auto_detect: bool = True):
        """Initialize query processor."""
        self.language_dim = language_dim
        self.spatial_dim = spatial_dim
        self.auto_detect = auto_detect
        
        # Initialize type-specific calculators
        self.language_calculator = WakeModeGeDIG(
            efficiency_weight=0.4,
            convergence_weight=0.6
        )
        
        self.spatial_calculator = WakeModeGeDIG(
            efficiency_weight=0.7,    # More efficiency for spatial
            convergence_weight=0.3
        )
        
        # Query type detection patterns
        self.spatial_keywords = {
            'move', 'go', 'navigate', 'position', 'location',
            'up', 'down', 'left', 'right', 'north', 'south',
            'maze', 'path', 'route', 'direction', 'wall',
            '移動', '方向', '位置', '迷路', '経路'
        }
        
        self.language_keywords = {
            'explain', 'describe', 'what', 'why', 'how',
            'meaning', 'define', 'understand', 'concept',
            '説明', '意味', '定義', '理解', '概念'
        }
        
    def detect_query_type(self, query_context: Union[str, Dict, QueryContext]) -> QueryType:
        """
        Automatically detect query type from context.
        
        Args:
            query_context: Query text, dict, or QueryContext object
            
        Returns:
            Detected QueryType
        """
        if isinstance(query_context, QueryContext):
            # Already has type
            if query_context.query_type != QueryType.UNKNOWN:
                return query_context.query_type
            text = query_context.query_text
        elif isinstance(query_context, dict):
            # Check for explicit type
            if 'query_type' in query_context:
                return QueryType(query_context['query_type'])
            text = query_context.get('query', '') or query_context.get('text', '')
        else:
            text = str(query_context)
            
        if not text:
            return QueryType.UNKNOWN
            
        # Convert to lowercase for matching
        text_lower = text.lower()
        
        # Count keyword matches
        spatial_count = sum(1 for kw in self.spatial_keywords if kw in text_lower)
        language_count = sum(1 for kw in self.language_keywords if kw in text_lower)
        
        # Decision logic
        if spatial_count > language_count:
            return QueryType.SPATIAL
        elif language_count > spatial_count:
            return QueryType.LANGUAGE
        elif spatial_count > 0 and language_count > 0:
            return QueryType.HYBRID
        else:
            # Additional heuristics
            if any(char in text for char in ['↑', '↓', '←', '→']):
                return QueryType.SPATIAL
            elif len(text.split()) > 5:  # Longer queries tend to be language
                return QueryType.LANGUAGE
            else:
                return QueryType.UNKNOWN
                
    def process_query(self,
                     graph: nx.Graph,
                     focal_nodes: Set[str],
                     query_context: Union[str, Dict, QueryContext]) -> WakeModeResult:
        """
        Process query with type-specific optimization.
        
        Args:
            graph: Knowledge graph
            focal_nodes: Query-relevant nodes
            query_context: Query information
            
        Returns:
            WakeModeResult with type-specific processing
        """
        # Ensure we have QueryContext object
        if not isinstance(query_context, QueryContext):
            context = self._create_query_context(query_context)
        else:
            context = query_context
            
        # Detect type if needed
        if self.auto_detect and context.query_type == QueryType.UNKNOWN:
            context.query_type = self.detect_query_type(context)
            
        logger.debug(f"Processing {context.query_type.value} query")
        
        # Route to appropriate processor
        if context.query_type == QueryType.SPATIAL:
            return self._process_spatial_query(graph, focal_nodes, context)
        elif context.query_type == QueryType.LANGUAGE:
            return self._process_language_query(graph, focal_nodes, context)
        elif context.query_type == QueryType.HYBRID:
            return self._process_hybrid_query(graph, focal_nodes, context)
        else:
            # Fallback to generic processing
            return self._process_generic_query(graph, focal_nodes, context)
            
    def _create_query_context(self, query_input: Union[str, Dict]) -> QueryContext:
        """Create QueryContext from various inputs."""
        if isinstance(query_input, str):
            return QueryContext(query_text=query_input)
        elif isinstance(query_input, dict):
            return QueryContext(**{k: v for k, v in query_input.items() 
                                 if k in QueryContext.__dataclass_fields__})
        else:
            return QueryContext(query_text=str(query_input))
            
    def _process_spatial_query(self,
                              graph: nx.Graph,
                              focal_nodes: Set[str],
                              context: QueryContext) -> WakeModeResult:
        """
        Process spatial/navigation queries with compact vectors.
        
        Optimizations:
        - Use 5D position vectors
        - Emphasize efficiency over pattern matching
        - Quick path finding
        """
        # Convert to spatial representation if needed
        spatial_graph = self._create_spatial_subgraph(graph, focal_nodes)
        
        # Add spatial patterns
        self._add_spatial_patterns(self.spatial_calculator)
        
        # Process with spatial optimization
        result = self.spatial_calculator.calculate_wake_mode_gedig(
            spatial_graph,
            focal_nodes,
            {
                'query_type': 'spatial',
                'position': context.position,
                'direction': context.direction,
                'bounds': context.spatial_bounds
            }
        )
        
        # Enhance with spatial-specific metrics
        result.metadata = {
            'vector_dimension': self.spatial_dim,
            'processing_mode': 'spatial_optimized',
            'path_efficiency': self._calculate_path_efficiency(spatial_graph, focal_nodes)
        }
        
        return result
        
    def _process_language_query(self,
                               graph: nx.Graph,
                               focal_nodes: Set[str],
                               context: QueryContext) -> WakeModeResult:
        """
        Process language queries with semantic vectors.
        
        Optimizations:
        - Use high-dimensional embeddings
        - Emphasize semantic similarity
        - Pattern matching over efficiency
        """
        # Ensure nodes have language embeddings
        if context.language_embedding is not None:
            self._attach_embeddings(graph, focal_nodes, context.language_embedding)
            
        # Add language patterns
        self._add_language_patterns(self.language_calculator)
        
        # Process with language optimization
        result = self.language_calculator.calculate_wake_mode_gedig(
            graph,
            focal_nodes,
            {
                'query_type': 'language',
                'text': context.query_text,
                'embedding': context.language_embedding
            }
        )
        
        # Enhance with language-specific metrics
        result.metadata = {
            'vector_dimension': self.language_dim,
            'processing_mode': 'semantic_optimized',
            'semantic_coherence': self._calculate_semantic_coherence(graph, focal_nodes)
        }
        
        return result
        
    def _process_hybrid_query(self,
                             graph: nx.Graph,
                             focal_nodes: Set[str],
                             context: QueryContext) -> WakeModeResult:
        """
        Process hybrid queries that combine spatial and language aspects.
        
        Strategy:
        - Run both processors
        - Combine results intelligently
        """
        # Process as spatial
        spatial_result = self._process_spatial_query(graph, focal_nodes, context)
        
        # Process as language
        language_result = self._process_language_query(graph, focal_nodes, context)
        
        # Combine results
        combined_gedig = (
            0.5 * spatial_result.gedig_value +
            0.5 * language_result.gedig_value
        )
        
        combined_result = WakeModeResult(
            gedig_value=combined_gedig,
            ged_value=(spatial_result.ged_value + language_result.ged_value) / 2,
            ig_value=(spatial_result.ig_value + language_result.ig_value) / 2,
            structural_improvement=(spatial_result.structural_improvement + 
                                  language_result.structural_improvement) / 2,
            information_integration=(spatial_result.information_integration +
                                   language_result.information_integration) / 2,
            pattern_match_score=max(spatial_result.pattern_match_score,
                                  language_result.pattern_match_score),
            convergence_score=(spatial_result.convergence_score +
                             language_result.convergence_score) / 2,
            efficiency_score=max(spatial_result.efficiency_score,
                               language_result.efficiency_score),
            mode="wake"
        )
        
        combined_result.metadata = {
            'processing_mode': 'hybrid',
            'spatial_component': spatial_result.gedig_value,
            'language_component': language_result.gedig_value
        }
        
        return combined_result
        
    def _process_generic_query(self,
                              graph: nx.Graph,
                              focal_nodes: Set[str],
                              context: QueryContext) -> WakeModeResult:
        """Fallback generic processing."""
        calculator = WakeModeGeDIG()
        return calculator.calculate_wake_mode_gedig(
            graph,
            focal_nodes,
            context.__dict__
        )
        
    def _create_spatial_subgraph(self,
                                graph: nx.Graph,
                                focal_nodes: Set[str]) -> nx.Graph:
        """Create spatial-optimized subgraph."""
        # Extract spatial components
        spatial_graph = nx.Graph()
        
        for node in graph.nodes():
            # Check if node has spatial attributes
            attrs = graph.nodes[node]
            if any(key in attrs for key in ['x', 'y', 'position', 'location']):
                spatial_graph.add_node(node, **attrs)
                
        # Copy relevant edges
        for u, v in graph.edges():
            if u in spatial_graph and v in spatial_graph:
                spatial_graph.add_edge(u, v, **graph.edges[u, v])
                
        return spatial_graph if len(spatial_graph) > 0 else graph
        
    def _attach_embeddings(self,
                          graph: nx.Graph,
                          focal_nodes: Set[str],
                          embedding: np.ndarray):
        """Attach language embeddings to nodes."""
        for node in focal_nodes:
            if node in graph:
                graph.nodes[node]['embedding'] = embedding
                
    def _add_spatial_patterns(self, calculator: WakeModeGeDIG):
        """Add common spatial navigation patterns."""
        # Direct path pattern
        direct_path = nx.Graph()
        direct_path.add_edges_from([('start', 'path'), ('path', 'goal')])
        calculator.add_pattern('direct_path', direct_path)
        
        # Wall avoidance pattern
        avoid_wall = nx.Graph()
        avoid_wall.add_edges_from([
            ('position', 'wall_check'),
            ('wall_check', 'alternative'),
            ('alternative', 'continue')
        ])
        calculator.add_pattern('wall_avoidance', avoid_wall)
        
    def _add_language_patterns(self, calculator: WakeModeGeDIG):
        """Add common language understanding patterns."""
        # Definition pattern
        definition = nx.Graph()
        definition.add_edges_from([
            ('concept', 'definition'),
            ('definition', 'example')
        ])
        calculator.add_pattern('definition', definition)
        
        # Explanation pattern
        explanation = nx.Graph()
        explanation.add_edges_from([
            ('question', 'context'),
            ('context', 'answer'),
            ('answer', 'justification')
        ])
        calculator.add_pattern('explanation', explanation)
        
    def _calculate_path_efficiency(self,
                                  graph: nx.Graph,
                                  focal_nodes: Set[str]) -> float:
        """Calculate spatial path efficiency metric."""
        if len(focal_nodes) < 2:
            return 1.0
            
        # Simple efficiency based on path length
        nodes = list(focal_nodes)
        try:
            if nx.has_path(graph, nodes[0], nodes[-1]):
                shortest = nx.shortest_path_length(graph, nodes[0], nodes[-1])
                actual = len(focal_nodes) - 1
                return min(1.0, actual / (shortest + 1))
        except:
            pass
            
        return 0.5
        
    def _calculate_semantic_coherence(self,
                                     graph: nx.Graph,
                                     focal_nodes: Set[str]) -> float:
        """Calculate semantic coherence metric."""
        # Simple coherence based on connectivity
        subgraph = graph.subgraph(focal_nodes)
        if len(focal_nodes) <= 1:
            return 1.0
            
        # Ratio of actual edges to possible edges
        actual_edges = subgraph.number_of_edges()
        possible_edges = len(focal_nodes) * (len(focal_nodes) - 1) / 2
        
        return actual_edges / possible_edges if possible_edges > 0 else 0.0