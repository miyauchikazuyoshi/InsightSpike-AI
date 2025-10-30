"""
Logarithmic sphere search for high-dimensional spaces.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from .sphere_search import NeighborNode, SimpleSphereSearch


@dataclass 
class LogarithmicSearchConfig:
    """Configuration for logarithmic radius mapping."""
    
    # Log-scale parameters (-inf to 0)
    log_radius: float = -1.0  # log(radius), e.g., -1 = radius 0.368
    log_inner_radius: float = -2.0  # For donut search
    
    # Alternative: percentile-based (0 to 100)
    percentile_radius: Optional[float] = None  # e.g., 50 = median distance
    
    # Alternative: similarity-based (0 to 1)
    similarity_threshold: Optional[float] = None  # e.g., 0.7 = cos_sim >= 0.7


class LogarithmicSphereSearch(SimpleSphereSearch):
    """
    Sphere search with logarithmic radius scaling for high dimensions.
    """
    
    def __init__(self, node_vectors: Dict[str, np.ndarray]):
        """Initialize with node vectors."""
        super().__init__(node_vectors)
        self._compute_distance_statistics()
    
    def _compute_distance_statistics(self):
        """Compute distance statistics for adaptive radius."""
        if len(self.node_vectors) < 2:
            return
            
        # Sample distances to understand distribution
        sample_size = min(1000, len(self.node_vectors))
        node_list = list(self.node_vectors.values())
        
        distances = []
        for i in range(sample_size):
            # Random pair
            idx1, idx2 = np.random.choice(len(node_list), 2, replace=False)
            dist = np.linalg.norm(node_list[idx1] - node_list[idx2])
            distances.append(dist)
        
        # Store statistics
        self.distance_stats = {
            'min': np.min(distances),
            'max': np.max(distances),
            'mean': np.mean(distances),
            'std': np.std(distances),
            'percentiles': {
                p: np.percentile(distances, p)
                for p in [10, 25, 50, 75, 90, 95, 99]
            }
        }
    
    def log_to_radius(self, log_radius: float) -> float:
        """
        Convert log-scale radius to actual radius.
        
        Args:
            log_radius: Logarithmic radius (typically -3 to 0)
                       -3 ≈ very tight (0.05)
                       -2 ≈ tight (0.135)  
                       -1 ≈ normal (0.368)
                       -0.5 ≈ broad (0.606)
                       0 ≈ all (1.0)
        
        Returns:
            Actual radius for distance calculation
        """
        return np.exp(log_radius)
    
    def similarity_to_radius(self, similarity: float) -> float:
        """
        Convert cosine similarity threshold to radius.
        
        Args:
            similarity: Desired minimum cosine similarity (0 to 1)
            
        Returns:
            Radius that corresponds to this similarity
        """
        # For normalized vectors: distance = sqrt(2 - 2*cos_sim)
        return np.sqrt(2 - 2 * similarity)
    
    def percentile_to_radius(self, percentile: float) -> float:
        """
        Convert percentile to radius based on distance distribution.
        
        Args:
            percentile: Percentile of distances (0 to 100)
            
        Returns:
            Radius at that percentile
        """
        if not hasattr(self, 'distance_stats'):
            # Fallback to fixed mapping
            return percentile / 100.0 * 1.4  # Rough approximation
        
        # Interpolate from computed percentiles
        percentiles = self.distance_stats['percentiles']
        
        # Find surrounding percentiles
        keys = sorted(percentiles.keys())
        for i, p in enumerate(keys):
            if p >= percentile:
                if i == 0:
                    return percentiles[p]
                # Linear interpolation
                p_low, p_high = keys[i-1], p
                r_low, r_high = percentiles[p_low], percentiles[p]
                
                t = (percentile - p_low) / (p_high - p_low)
                return r_low + t * (r_high - r_low)
        
        return percentiles[keys[-1]]  # Max percentile
    
    def search_log_sphere(
        self,
        query_vec: np.ndarray,
        log_radius: float = -1.0,
        max_neighbors: Optional[int] = None
    ) -> List[NeighborNode]:
        """
        Search using logarithmic radius.
        
        Args:
            query_vec: Query vector
            log_radius: Log of radius (e.g., -1 for e^-1 ≈ 0.368)
            max_neighbors: Maximum neighbors
            
        Returns:
            List of neighbors
        """
        radius = self.log_to_radius(log_radius)
        return self.search_sphere(query_vec, radius, max_neighbors)
    
    def search_similarity_sphere(
        self,
        query_vec: np.ndarray,
        min_similarity: float = 0.7,
        max_neighbors: Optional[int] = None
    ) -> List[NeighborNode]:
        """
        Search based on cosine similarity threshold.
        
        Args:
            query_vec: Query vector
            min_similarity: Minimum cosine similarity (0 to 1)
            max_neighbors: Maximum neighbors
            
        Returns:
            List of neighbors with at least min_similarity
        """
        radius = self.similarity_to_radius(min_similarity)
        return self.search_sphere(query_vec, radius, max_neighbors)
    
    def search_percentile_sphere(
        self,
        query_vec: np.ndarray,
        percentile: float = 50.0,
        max_neighbors: Optional[int] = None
    ) -> List[NeighborNode]:
        """
        Search using percentile-based radius.
        
        Args:
            query_vec: Query vector  
            percentile: Distance percentile (0-100)
            max_neighbors: Maximum neighbors
            
        Returns:
            List of neighbors within percentile radius
        """
        radius = self.percentile_to_radius(percentile)
        return self.search_sphere(query_vec, radius, max_neighbors)
    
    def search_adaptive_donut(
        self,
        query_vec: np.ndarray,
        log_inner: float = -2.0,
        log_outer: float = -0.5,
        max_neighbors: Optional[int] = None
    ) -> List[NeighborNode]:
        """
        Donut search with logarithmic radii.
        
        Args:
            query_vec: Query vector
            log_inner: Log of inner radius (e.g., -2)
            log_outer: Log of outer radius (e.g., -0.5)
            max_neighbors: Maximum neighbors
            
        Returns:
            List of neighbors in donut region
        """
        inner_radius = self.log_to_radius(log_inner)
        outer_radius = self.log_to_radius(log_outer)
        return self.search_donut(query_vec, inner_radius, outer_radius, max_neighbors)
    
    def get_radius_recommendations(self) -> Dict[str, Dict[str, float]]:
        """
        Get recommended radius values based on data distribution.
        
        Returns:
            Dictionary of recommendations
        """
        if not hasattr(self, 'distance_stats'):
            return {
                'note': 'No statistics available',
                'defaults': {
                    'tight': 0.3,
                    'normal': 0.6,
                    'broad': 0.9
                }
            }
        
        stats = self.distance_stats
        
        return {
            'statistics': {
                'mean_distance': stats['mean'],
                'std_distance': stats['std'],
                'typical_range': (stats['percentiles'][25], stats['percentiles'][75])
            },
            'log_scale': {
                'very_tight': -3.0,   # ~5th percentile
                'tight': -2.0,        # ~15th percentile  
                'normal': -1.0,       # ~35th percentile
                'broad': -0.5,        # ~60th percentile
                'very_broad': 0.0     # ~100th percentile
            },
            'similarity_based': {
                'high': 0.8,          # cos_sim >= 0.8
                'medium': 0.6,        # cos_sim >= 0.6
                'low': 0.4            # cos_sim >= 0.4
            },
            'percentile_based': {
                'selective': 25,      # Bottom quarter
                'balanced': 50,       # Median
                'inclusive': 75       # Top three quarters
            }
        }


def create_intuitive_searcher(node_vectors: Dict[str, np.ndarray]) -> LogarithmicSphereSearch:
    """
    Factory function to create searcher with intuitive interface.
    
    Args:
        node_vectors: Dictionary of vectors
        
    Returns:
        Configured logarithmic searcher
    """
    searcher = LogarithmicSphereSearch(node_vectors)
    
    # Add convenience methods
    def search_tight(query, k=10):
        return searcher.search_log_sphere(query, log_radius=-2.0, max_neighbors=k)
    
    def search_normal(query, k=20):
        return searcher.search_log_sphere(query, log_radius=-1.0, max_neighbors=k)
    
    def search_broad(query, k=30):
        return searcher.search_log_sphere(query, log_radius=-0.5, max_neighbors=k)
    
    # Attach methods
    searcher.search_tight = search_tight
    searcher.search_normal = search_normal
    searcher.search_broad = search_broad
    
    return searcher