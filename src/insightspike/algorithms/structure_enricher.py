
"""
Structure Enricher Module
========================

Provides a decoupled mechanism to enrich graph nodes with structural features/roles.
Separates the "write" side effects (updating node attributes) from the pure metric calculation.
"""

import logging
from typing import Any, Dict, List, Optional, Set

import networkx as nx
import numpy as np

from ..config.models import StructuralSimilarityConfig
from .structural_similarity import StructuralSimilarityEvaluator

logger = logging.getLogger(__name__)


class StructureEnricher:
    """
    Enriches graph nodes with structural metadata.
    
    This class is responsible for computing structural signatures or roles
    (e.g., 'hub', 'spoke', 'bridge') and persisting them as node attributes.
    """
    
    def __init__(self, config: Optional[StructuralSimilarityConfig] = None):
        """
        Initialize the enricher.
        
        Args:
            config: Configuration for structural analysis.
        """
        self.config = config or StructuralSimilarityConfig(enabled=True)
        self.evaluator = StructuralSimilarityEvaluator(self.config)
        
    def enrich_graph(
        self, 
        graph: nx.Graph, 
        center_node: Optional[str] = None,
        radius: int = 2
    ) -> nx.Graph:
        """
        Compute and attach structural features to nodes in the graph.
        
        Args:
            graph: The NetworkX graph to enrich (modified in-place).
            center_node: Optional focal point. If provided, enrichment focuses 
                         on the neighborhood of this node.
            radius: Radius around center_node to enrich.
            
        Returns:
            The modified graph (for chaining).
        """
        nodes_to_process = self._get_nodes_to_process(graph, center_node, radius)
        
        logger.debug(f"Enriching {len(nodes_to_process)} nodes with structural features.")
        
        for node in nodes_to_process:
            self._enrich_node(graph, node)
            
        return graph
        
    def _get_nodes_to_process(
        self, 
        graph: nx.Graph, 
        center: Optional[str], 
        radius: int
    ) -> Set[str]:
        """Determine which nodes need enrichment update."""
        if center is None or center not in graph:
            # If graph is small enough, process all. Otherwise, warn/limit?
            # For now, process all nodes (safe for typical local subgraphs)
            return set(graph.nodes())
        
        # Get ego graph nodes
        try:
            sub = nx.ego_graph(graph, center, radius=radius)
            return set(sub.nodes())
        except Exception:
            return {center}

    def _enrich_node(self, graph: nx.Graph, node: str) -> None:
        """Calculate and attach features for a single node."""
        
        # 1. Motif-based Structural Signature (Vector)
        # Keep role heuristics aligned with motif signature ordering.
        
        try:
            signature = self.evaluator.extract_motif_signature(graph, center=node)
            
            # Convert to list for JSON serialization compatibility if needed
            sig_list = signature.tolist() if hasattr(signature, 'tolist') else list(signature)
            
            # 2. Determine Role (Heuristic)
            role = self._classify_role(signature)
            
            # 3. Update Node Attributes
            graph.nodes[node]['structure_signature'] = sig_list
            graph.nodes[node]['structure_role'] = role
            
            # Optional: Hub score from signature
            # Index 2 is hub_ratio in motif signature ordering
            if len(signature) > 2:
                graph.nodes[node]['hub_score'] = float(signature[2])
                
        except Exception as e:
            logger.warning(f"Failed to enrich node {node}: {e}")

    def _classify_role(self, signature: np.ndarray) -> str:
        """
        Classify node role based on signature vector.
        
        Signature indices (from structural_similarity.py motif signature):
        0: triangles
        1: stars
        2: hub_ratio
        3: chains
        4: squares
        ...
        """
        if len(signature) < 5:
            return "unknown"
            
        triangles = signature[0]
        stars = signature[1]
        hub_ratio = signature[2]
        chains = signature[3]
        
        # Simple rule-based classification
        if hub_ratio > 2.0 or stars > 0.5:
            return "hub"
        if triangles > 0.5:
            return "cluster_core"
        if chains > 0.5:
            return "bridge_or_chain"
            
        return "peripheral"

    def get_node_structural_features(self, graph: nx.Graph, node: str) -> Dict[str, Any]:
        """Retrieve stored structural features for a node."""
        if node not in graph:
            return {}
        
        data = graph.nodes[node]
        return {
            "signature": data.get("structure_signature"),
            "role": data.get("structure_role"),
            "hub_score": data.get("hub_score", 0.0)
        }
