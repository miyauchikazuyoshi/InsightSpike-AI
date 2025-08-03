"""
Dimension-aware logarithmic sphere search that preserves 3D intuition.
"""

import numpy as np
from typing import List, Dict, Optional
from .sphere_search import NeighborNode, SimpleSphereSearch


class DimensionAwareSphereSearch(SimpleSphereSearch):
    """
    Sphere search that uses log_n(x) where n is the dimension.
    This preserves 3D volume intuition in high-dimensional spaces.
    """
    
    def __init__(self, node_vectors: Dict[str, np.ndarray]):
        """Initialize with node vectors and compute dimension."""
        super().__init__(node_vectors)
        
        # Get dimension from vectors
        if node_vectors:
            first_vec = next(iter(node_vectors.values()))
            self.dimension = len(first_vec)
        else:
            self.dimension = 768  # Default for typical embeddings
        
        # Precompute log base
        self.log_base = self.dimension
        
    def volume_fraction_to_radius(self, volume_fraction: float) -> float:
        """
        Convert desired volume fraction to actual radius.
        
        In 3D: radius 0.5 → volume fraction = 0.5^3 = 0.125 (1/8)
        In n-D: radius r → volume fraction = r^n
        
        To get same intuition: r = volume_fraction^(1/n)
        
        Args:
            volume_fraction: Desired fraction of unit ball volume (0 to 1)
                           0.125 = 1/8 (like 3D radius 0.5)
                           0.5 = 1/2 (like 3D radius ~0.794)
                           
        Returns:
            Radius that gives this volume fraction in n dimensions
        """
        return np.power(volume_fraction, 1.0 / self.dimension)
    
    def intuitive_radius_to_actual(self, intuitive_radius: float) -> float:
        """
        Convert 3D-intuitive radius to actual radius for n dimensions.
        
        This preserves the volume relationship from 3D:
        - intuitive 0.5 → 1/8 of space (like in 3D)
        - intuitive 0.25 → 1/64 of space (like in 3D)
        
        Args:
            intuitive_radius: Radius as if in 3D (0 to 1)
            
        Returns:
            Actual radius for n-dimensional space
        """
        # Volume fraction in 3D
        volume_fraction_3d = intuitive_radius ** 3
        
        # Convert to n-D radius
        return self.volume_fraction_to_radius(volume_fraction_3d)
    
    def log_dimension_radius(self, log_fraction: float) -> float:
        """
        Use log base n (dimension) for radius calculation.
        
        log_n(volume_fraction) = log_fraction
        volume_fraction = n^log_fraction
        radius = volume_fraction^(1/n) = n^(log_fraction/n)
        
        Args:
            log_fraction: Log base n of desired volume fraction
                         -1 → volume = 1/n of total
                         -2 → volume = 1/n² of total
                         
        Returns:
            Actual radius
        """
        return np.power(self.dimension, log_fraction / self.dimension)
    
    def search_intuitive(
        self,
        query_vec: np.ndarray,
        intuitive_radius: float = 0.5,
        max_neighbors: Optional[int] = None
    ) -> List[NeighborNode]:
        """
        Search using 3D-intuitive radius.
        
        Args:
            query_vec: Query vector
            intuitive_radius: Radius as if in 3D space (0.5 = half radius)
            max_neighbors: Maximum neighbors
            
        Returns:
            Neighbors within intuitive radius
        """
        actual_radius = self.intuitive_radius_to_actual(intuitive_radius)
        return self.search_sphere(query_vec, actual_radius, max_neighbors)
    
    def search_volume_fraction(
        self,
        query_vec: np.ndarray,
        volume_fraction: float = 0.125,  # 1/8
        max_neighbors: Optional[int] = None
    ) -> List[NeighborNode]:
        """
        Search by volume fraction.
        
        Args:
            query_vec: Query vector
            volume_fraction: Fraction of unit ball to search (0.125 = 1/8)
            max_neighbors: Maximum neighbors
            
        Returns:
            Neighbors within volume fraction
        """
        radius = self.volume_fraction_to_radius(volume_fraction)
        return self.search_sphere(query_vec, radius, max_neighbors)
    
    def search_log_dimension(
        self,
        query_vec: np.ndarray,
        log_fraction: float = -1.0,
        max_neighbors: Optional[int] = None
    ) -> List[NeighborNode]:
        """
        Search using log base dimension.
        
        Args:
            query_vec: Query vector
            log_fraction: log_n(volume_fraction) where n = dimension
            max_neighbors: Maximum neighbors
            
        Returns:
            Neighbors within radius
        """
        radius = self.log_dimension_radius(log_fraction)
        return self.search_sphere(query_vec, radius, max_neighbors)
    
    def get_radius_mapping(self) -> Dict[str, Dict[str, float]]:
        """
        Get radius mappings for current dimension.
        
        Returns:
            Dictionary showing different radius calculations
        """
        return {
            'dimension': self.dimension,
            'intuitive_to_actual': {
                '0.25 (3D quarter)': self.intuitive_radius_to_actual(0.25),
                '0.5 (3D half)': self.intuitive_radius_to_actual(0.5),
                '0.75 (3D three-quarter)': self.intuitive_radius_to_actual(0.75),
            },
            'volume_fraction_to_radius': {
                '1/64': self.volume_fraction_to_radius(1/64),
                '1/8': self.volume_fraction_to_radius(1/8),
                '1/2': self.volume_fraction_to_radius(1/2),
            },
            'log_dimension_to_radius': {
                'log_n(-2)': self.log_dimension_radius(-2),
                'log_n(-1)': self.log_dimension_radius(-1),
                'log_n(-0.5)': self.log_dimension_radius(-0.5),
            }
        }
    
    def adaptive_donut_search(
        self,
        query_vec: np.ndarray,
        inner_volume_fraction: float = 0.015625,  # 1/64 (like 3D r=0.25)
        outer_volume_fraction: float = 0.125,      # 1/8 (like 3D r=0.5)
        max_neighbors: Optional[int] = None
    ) -> List[NeighborNode]:
        """
        Donut search with volume-based radii.
        
        Args:
            query_vec: Query vector
            inner_volume_fraction: Inner volume exclusion
            outer_volume_fraction: Outer volume inclusion
            max_neighbors: Maximum neighbors
            
        Returns:
            Neighbors in donut region
        """
        inner_radius = self.volume_fraction_to_radius(inner_volume_fraction)
        outer_radius = self.volume_fraction_to_radius(outer_volume_fraction)
        return self.search_donut(query_vec, inner_radius, outer_radius, max_neighbors)


def demonstrate_dimension_scaling():
    """
    Show how radius scales with dimension to preserve volume intuition.
    """
    dimensions = [3, 10, 50, 100, 384, 768]
    intuitive_radius = 0.5  # "Half" radius in 3D
    
    print(f"Intuitive radius: {intuitive_radius} (like 3D)")
    print(f"This gives volume fraction: {intuitive_radius**3:.3f} in 3D\n")
    
    print("Actual radius needed in different dimensions:")
    print("Dim | Actual Radius | Check: radius^dim")
    print("-" * 40)
    
    for dim in dimensions:
        volume_fraction = intuitive_radius ** 3  # 1/8
        actual_radius = np.power(volume_fraction, 1.0 / dim)
        check = actual_radius ** dim
        
        print(f"{dim:3d} | {actual_radius:.6f} | {check:.6f}")
    
    print("\nNote: In all dimensions, we get the same volume fraction!")


if __name__ == "__main__":
    demonstrate_dimension_scaling()