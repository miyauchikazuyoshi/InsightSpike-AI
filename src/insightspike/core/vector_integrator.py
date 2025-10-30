"""
Vector Integration Module
========================

Unified vector integration for various use cases:
- Insight vector generation
- Episode branching
- Message passing
- Context merging
"""

import numpy as np
from typing import List, Optional, Dict, Union, Literal
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


# Integration configurations
INTEGRATION_CONFIGS = {
    "insight": {
        "primary_weight": 1.0,      # Query vector weight
        "secondary_weights": "similarity",  # Document weights based on similarity
        "aggregation": "weighted_mean",
        "normalize": True
    },
    "episode_branching": {
        "primary_weight": 0.4,      # Parent episode weight
        "secondary_weights": "equal",  # Neighbor weights equally distributed
        "aggregation": "weighted_mean", 
        "normalize": True
    },
    "message_passing": {
        "primary_weight": None,     # Determined by self_loop_weight
        "secondary_weights": "custom",  # Provided by caller
        "aggregation": "weighted_mean",
        "normalize": False
    },
    "context_merging": {
        "primary_weight": 0.5,
        "secondary_weights": "distance",  # Based on semantic distance
        "aggregation": "weighted_mean",
        "normalize": True
    }
}


class VectorIntegrator:
    """
    Unified vector integration for various InsightSpike operations.
    
    This class provides a consistent interface for combining multiple vectors
    into a single representation, used across insight generation, episode
    branching, and message passing.
    """
    
    def __init__(self, weight_manager=None):
        """Initialize the vector integrator.
        
        Args:
            weight_manager: Optional WeightVectorManager for dimension weighting
        """
        self.configs = INTEGRATION_CONFIGS.copy()
        self.weight_manager = weight_manager
    
    def integrate_vectors(self,
                         vectors: List[np.ndarray],
                         primary_vector: Optional[np.ndarray] = None,
                         integration_type: str = "weighted_mean",
                         custom_weights: Optional[List[float]] = None,
                         config_overrides: Optional[Dict] = None) -> np.ndarray:
        """
        Integrate multiple vectors into a single vector.
        
        Args:
            vectors: List of vectors to integrate
            primary_vector: Primary vector (e.g., query, parent episode)
            integration_type: Type of integration ("insight", "episode_branching", etc.)
            custom_weights: Custom weights for vectors (overrides config)
            config_overrides: Override default configuration
            
        Returns:
            Integrated vector
        """
        if not vectors and primary_vector is None:
            raise ValueError("No vectors provided for integration")
        
        # Get configuration
        config = self._get_config(integration_type, config_overrides)
        
        # Prepare all vectors
        all_vectors = []
        weights = []
        
        # Add primary vector if provided
        if primary_vector is not None:
            primary_weight = config.get("primary_weight", 1.0)
            # Only add primary vector if it has a weight
            if primary_weight is not None:
                all_vectors.append(primary_vector)
                weights.append(primary_weight)
            elif integration_type != "message_passing":
                # For non-message-passing types, always include primary
                all_vectors.append(primary_vector)
                weights.append(1.0)
        
        # Add secondary vectors
        if vectors:
            # Apply dimension weights if manager is available
            if self.weight_manager:
                vectors = [self.weight_manager.apply_weights(v) for v in vectors]
            
            all_vectors.extend(vectors)
            
            # Determine secondary weights
            if custom_weights is not None:
                weights.extend(custom_weights)
            else:
                secondary_weights = self._calculate_secondary_weights(
                    vectors, primary_vector, config
                )
                weights.extend(secondary_weights)
        
        # Normalize weights if needed
        if weights:
            weights = np.array(weights, dtype=float)
            total = weights.sum()
            if total > 0:
                weights = weights / total
        else:
            # No weights provided, use None for aggregation
            weights = None
        
        # Perform aggregation
        result = self._aggregate(all_vectors, weights, config["aggregation"])
        
        # Normalize result if configured
        if config.get("normalize", True):
            norm = np.linalg.norm(result)
            if norm > 0:
                result = result / norm
        
        return result
    
    def _get_config(self, integration_type: str, overrides: Optional[Dict]) -> Dict:
        """Get configuration for integration type with overrides."""
        # Use predefined config or default
        if integration_type in self.configs:
            config = self.configs[integration_type].copy()
        else:
            # Default configuration
            config = {
                "primary_weight": 0.5,
                "secondary_weights": "equal",
                "aggregation": "weighted_mean",
                "normalize": True
            }
        
        # Apply overrides
        if overrides:
            config.update(overrides)
        
        return config
    
    def _calculate_secondary_weights(self,
                                   vectors: List[np.ndarray],
                                   primary_vector: Optional[np.ndarray],
                                   config: Dict) -> List[float]:
        """Calculate weights for secondary vectors based on configuration."""
        weight_type = config["secondary_weights"]
        
        if weight_type == "equal":
            # Equal weights
            return [1.0 / len(vectors)] * len(vectors)
        
        elif weight_type == "similarity" and primary_vector is not None:
            # Weights based on similarity to primary vector
            vectors_array = np.array(vectors)
            primary_2d = primary_vector.reshape(1, -1)
            similarities = cosine_similarity(vectors_array, primary_2d).flatten()
            
            # Ensure non-negative
            similarities = np.maximum(similarities, 0)
            
            # Normalize to sum to 1
            if similarities.sum() > 0:
                return similarities / similarities.sum()
            else:
                return [1.0 / len(vectors)] * len(vectors)
        
        elif weight_type == "distance" and primary_vector is not None:
            # Weights based on inverse distance
            vectors_array = np.array(vectors)
            distances = np.array([
                np.linalg.norm(v - primary_vector) for v in vectors
            ])
            
            # Inverse distance weights
            epsilon = 1e-8
            weights = 1.0 / (distances + epsilon)
            
            # Normalize
            return weights / weights.sum()
        
        else:
            # Default to equal weights
            return [1.0 / len(vectors)] * len(vectors)
    
    def _aggregate(self,
                   vectors: List[np.ndarray],
                   weights: Optional[np.ndarray],
                   method: str) -> np.ndarray:
        """Aggregate vectors using specified method."""
        vectors_array = np.array(vectors)
        
        if method == "weighted_mean":
            if weights is not None:
                return np.average(vectors_array, axis=0, weights=weights)
            else:
                return np.mean(vectors_array, axis=0)
        
        elif method == "mean":
            return np.mean(vectors_array, axis=0)
        
        elif method == "max":
            return np.max(vectors_array, axis=0)
        
        elif method == "sum":
            # Raw sum semantics for tests (ignore weights)
            return np.sum(vectors_array, axis=0)
        
        else:
            # Default to weighted mean
            if weights is not None:
                return np.average(vectors_array, axis=0, weights=weights)
            else:
                return np.mean(vectors_array, axis=0)
    
    def create_insight_vector(self,
                            document_embeddings: List[np.ndarray],
                            query_vector: np.ndarray) -> np.ndarray:
        """
        Convenience method for insight vector creation.
        
        Args:
            document_embeddings: List of document embeddings
            query_vector: Query embedding
            
        Returns:
            Insight vector
        """
        return self.integrate_vectors(
            document_embeddings,
            primary_vector=query_vector,
            integration_type="insight"
        )
    
    def create_branch_vector(self,
                           parent_vector: np.ndarray,
                           neighbor_vectors: List[np.ndarray]) -> np.ndarray:
        """
        Convenience method for episode branching.
        
        Args:
            parent_vector: Parent episode vector
            neighbor_vectors: Neighbor episode vectors
            
        Returns:
            Branch vector
        """
        return self.integrate_vectors(
            neighbor_vectors,
            primary_vector=parent_vector,
            integration_type="episode_branching"
        )
    
    def update_node_representation(self,
                                 node_vector: np.ndarray,
                                 neighbor_vectors: List[np.ndarray],
                                 weights: List[float]) -> np.ndarray:
        """
        Convenience method for message passing node updates.
        
        Args:
            node_vector: Current node representation
            neighbor_vectors: Neighbor representations
            weights: Message weights
            
        Returns:
            Updated node representation
        """
        return self.integrate_vectors(
            neighbor_vectors,
            primary_vector=node_vector,
            integration_type="message_passing",
            custom_weights=weights
        )