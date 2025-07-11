"""
Unified Entropy Calculator with Content/Structure Separation
===========================================================

Implements proper separation of content and structural entropy
as recommended in the Layer 3 improvements review.

This module provides a unified interface for calculating:
- Content entropy: Based on node features/embeddings
- Structural entropy: Based on graph topology
- Combined entropy: Weighted combination of both
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

# Try to import optional dependencies
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.getLogger(__name__).warning("sklearn not available, will use numpy fallback for cosine similarity")

from .information_gain import InformationGain, EntropyMethod
from .structural_entropy import (
    degree_distribution_entropy,
    von_neumann_entropy,
    structural_entropy as calc_structural_entropy,
    clustering_coefficient_entropy
)

logger = logging.getLogger(__name__)

__all__ = ["EntropyCalculator", "EntropyResult", "ContentStructureSeparation"]


@dataclass
class EntropyResult:
    """Result of entropy calculation with content/structure separation."""
    content_entropy: float
    structural_entropy: float
    combined_entropy: float
    content_weight: float
    structure_weight: float
    method_used: str
    
    @property
    def dominant_component(self) -> str:
        """Which component contributes more to combined entropy."""
        content_contrib = self.content_entropy * self.content_weight
        structure_contrib = self.structural_entropy * self.structure_weight
        return "content" if content_contrib > structure_contrib else "structure"
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            "content_entropy": self.content_entropy,
            "structural_entropy": self.structural_entropy,
            "combined_entropy": self.combined_entropy,
            "content_weight": self.content_weight,
            "structure_weight": self.structure_weight,
            "dominant": self.dominant_component
        }


class ContentStructureSeparation:
    """Separates content and structural aspects of data for entropy calculation."""
    
    @staticmethod
    def extract_content(data: Any) -> Optional[np.ndarray]:
        """Extract content features from data."""
        # PyTorch Geometric Data
        if hasattr(data, 'x') and hasattr(data.x, 'numpy'):
            return data.x.cpu().numpy()
        elif hasattr(data, 'x') and isinstance(data.x, np.ndarray):
            return data.x
        
        # NetworkX graph with node features
        if hasattr(data, 'nodes') and hasattr(data, 'edges'):
            try:
                import networkx as nx
                if isinstance(data, nx.Graph):
                    # Try to extract node features
                    features = []
                    for node in data.nodes():
                        if 'features' in data.nodes[node]:
                            features.append(data.nodes[node]['features'])
                        elif 'embedding' in data.nodes[node]:
                            features.append(data.nodes[node]['embedding'])
                    
                    if features:
                        return np.array(features)
            except:
                pass
        
        # Raw numpy array
        if isinstance(data, np.ndarray):
            return data
        
        # List of vectors
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], (list, np.ndarray)):
                return np.array(data)
        
        return None
    
    @staticmethod
    def extract_structure(data: Any) -> Any:
        """Extract structural graph from data."""
        # Already a graph structure
        if hasattr(data, 'edge_index') or hasattr(data, 'edges'):
            return data
        
        # Try to build graph from similarity
        content = ContentStructureSeparation.extract_content(data)
        if content is not None and len(content) > 1:
            try:
                import networkx as nx
                if SKLEARN_AVAILABLE:
                    # Build similarity graph using sklearn
                    sim_matrix = cosine_similarity(content)
                else:
                    # Fallback to manual cosine similarity
                    norms = np.linalg.norm(content, axis=1, keepdims=True)
                    normalized = content / (norms + 1e-8)
                    sim_matrix = np.dot(normalized, normalized.T)
                
                threshold = np.percentile(sim_matrix.flatten(), 75)  # Top 25% similarities
                
                G = nx.Graph()
                n_nodes = len(content)
                G.add_nodes_from(range(n_nodes))
                
                for i in range(n_nodes):
                    for j in range(i+1, n_nodes):
                        if sim_matrix[i, j] > threshold:
                            G.add_edge(i, j, weight=sim_matrix[i, j])
                
                return G
            except Exception as e:
                logger.warning(f"Failed to build structure from content: {e}")
        
        return None


class EntropyCalculator:
    """
    Unified entropy calculator with proper content/structure separation.
    
    This class implements the recommendation from the Layer 3 review
    to clearly separate content-based and structure-based entropy.
    """
    
    def __init__(self,
                 content_method: Union[str, EntropyMethod] = EntropyMethod.CLUSTERING,
                 structure_method: str = "combined",
                 content_weight: float = 0.6,
                 structure_weight: float = 0.4):
        """
        Initialize entropy calculator.
        
        Args:
            content_method: Method for content entropy (from InformationGain)
            structure_method: Method for structural entropy
            content_weight: Weight for content entropy in combination
            structure_weight: Weight for structural entropy in combination
        """
        self.content_calculator = InformationGain(method=content_method)
        self.structure_method = structure_method
        
        # Normalize weights
        total_weight = content_weight + structure_weight
        self.content_weight = content_weight / total_weight
        self.structure_weight = structure_weight / total_weight
        
        self.separator = ContentStructureSeparation()
    
    def calculate_entropy(self, data: Any) -> EntropyResult:
        """
        Calculate entropy with content/structure separation.
        
        Args:
            data: Input data (graph, embeddings, or mixed)
            
        Returns:
            EntropyResult with separated components
        """
        # Extract content and structure
        content = self.separator.extract_content(data)
        structure = self.separator.extract_structure(data)
        
        # Calculate content entropy
        content_entropy = 0.0
        if content is not None:
            try:
                content_entropy = self.content_calculator._calculate_entropy(content)
            except Exception as e:
                logger.warning(f"Content entropy calculation failed: {e}")
        
        # Calculate structural entropy
        structural_entropy = 0.0
        if structure is not None:
            try:
                if self.structure_method == "combined":
                    measures = calc_structural_entropy(structure)
                    structural_entropy = measures.get("combined", 0.0)
                elif self.structure_method == "degree":
                    structural_entropy = degree_distribution_entropy(structure)
                elif self.structure_method == "von_neumann":
                    structural_entropy = von_neumann_entropy(structure)
                elif self.structure_method == "clustering":
                    structural_entropy = clustering_coefficient_entropy(structure)
                else:
                    logger.warning(f"Unknown structure method: {self.structure_method}")
            except Exception as e:
                logger.warning(f"Structural entropy calculation failed: {e}")
        
        # Calculate combined entropy
        combined_entropy = (
            self.content_weight * content_entropy +
            self.structure_weight * structural_entropy
        )
        
        return EntropyResult(
            content_entropy=content_entropy,
            structural_entropy=structural_entropy,
            combined_entropy=combined_entropy,
            content_weight=self.content_weight,
            structure_weight=self.structure_weight,
            method_used=f"{self.content_calculator.method.value}+{self.structure_method}"
        )
    
    def calculate_delta_entropy(self, 
                               data_before: Any, 
                               data_after: Any) -> Tuple[float, EntropyResult, EntropyResult]:
        """
        Calculate change in entropy between two states.
        
        Args:
            data_before: Initial state
            data_after: Final state
            
        Returns:
            Tuple of (delta_entropy, before_result, after_result)
            where delta_entropy is before - after (positive = information gain)
        """
        before_result = self.calculate_entropy(data_before)
        after_result = self.calculate_entropy(data_after)
        
        # Positive delta means entropy decreased (information gained)
        delta = before_result.combined_entropy - after_result.combined_entropy
        
        return delta, before_result, after_result
    
    def calculate_insight_score(self,
                               data_before: Any,
                               data_after: Any,
                               normalize: bool = True) -> Dict[str, float]:
        """
        Calculate insight score based on entropy changes.
        
        Args:
            data_before: Initial state
            data_after: Final state
            normalize: Whether to normalize by initial entropy
            
        Returns:
            Dictionary with insight scores
        """
        delta, before, after = self.calculate_delta_entropy(data_before, data_after)
        
        # Calculate component-wise changes
        delta_content = before.content_entropy - after.content_entropy
        delta_structure = before.structural_entropy - after.structural_entropy
        
        # Normalize if requested
        if normalize:
            if before.content_entropy > 0:
                delta_content /= before.content_entropy
            if before.structural_entropy > 0:
                delta_structure /= before.structural_entropy
            if before.combined_entropy > 0:
                delta /= before.combined_entropy
        
        return {
            "total_insight": delta,
            "content_insight": delta_content,
            "structure_insight": delta_structure,
            "content_contribution": delta_content * self.content_weight,
            "structure_contribution": delta_structure * self.structure_weight,
            "insight_type": "content" if abs(delta_content) > abs(delta_structure) else "structure"
        }
    
    def set_weights(self, content_weight: float, structure_weight: float):
        """Update the weights for combining content and structural entropy."""
        total = content_weight + structure_weight
        if total > 0:
            self.content_weight = content_weight / total
            self.structure_weight = structure_weight / total
        else:
            self.content_weight = 0.5
            self.structure_weight = 0.5