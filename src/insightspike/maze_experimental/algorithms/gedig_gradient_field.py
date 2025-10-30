"""
geDIG Gradient Field
====================

Define a gradient field in vector space where each point has a geDIG gradient.
Instead of searching nearby nodes, follow the gradient to find stable points.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Tuple, Optional, List
import numpy as np


class GeDIGGradientField:
    """
    Define geDIG as a continuous field in vector space
    """
    
    def __init__(self, graph: Data, w1: float = 1.0, kT: float = 0.1):
        """
        Initialize gradient field from graph
        
        Args:
            graph: Current knowledge graph
            w1: Weight for GED component
            kT: Temperature for IG component
        """
        self.graph = graph
        self.w1 = w1
        self.kT = kT
        
        # Precompute graph properties for efficiency
        self._precompute_graph_properties()
        
    def _precompute_graph_properties(self):
        """Precompute expensive graph properties"""
        # Node feature matrix
        self.node_features = self.graph.x
        self.num_nodes = self.graph.num_nodes
        
        # Compute feature covariance for Mahalanobis distance
        if self.num_nodes > 1:
            self.feature_mean = self.node_features.mean(dim=0)
            centered = self.node_features - self.feature_mean
            self.feature_cov = (centered.T @ centered) / (self.num_nodes - 1)
            # Add small diagonal for numerical stability
            self.feature_cov += torch.eye(self.feature_cov.size(0)) * 1e-6
            self.feature_cov_inv = torch.inverse(self.feature_cov)
        else:
            self.feature_mean = self.node_features[0]
            self.feature_cov_inv = torch.eye(self.node_features.size(1))
    
    def gedig_field(self, point: torch.Tensor) -> torch.Tensor:
        """
        Compute geDIG value at any point in vector space
        
        This defines a scalar field over the vector space
        """
        # GED component: Distance from graph manifold
        ged = self._ged_from_manifold(point)
        
        # IG component: Information gain potential
        ig = self._information_potential(point)
        
        # Combined geDIG field value
        # F = w1*GED - kT*IG (we want to minimize this)
        return self.w1 * ged - self.kT * ig
    
    def gradient(self, point: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of geDIG field at a point
        
        Returns the direction of steepest increase
        (negate for descent direction)
        """
        point = point.requires_grad_(True)
        
        # Compute field value
        field_value = self.gedig_field(point)
        
        # Compute gradient via autograd
        field_value.backward()
        grad = point.grad.detach()
        
        return grad
    
    def _ged_from_manifold(self, point: torch.Tensor) -> torch.Tensor:
        """
        Approximate GED as distance from graph manifold
        
        Uses a smooth approximation based on:
        1. Mahalanobis distance from feature distribution
        2. Minimum distance to existing nodes (soft-min)
        """
        # Mahalanobis distance from graph distribution
        diff = point - self.feature_mean
        mahalanobis = torch.sqrt(diff @ self.feature_cov_inv @ diff)
        
        # Soft minimum distance to nodes
        distances = torch.norm(self.node_features - point.unsqueeze(0), dim=1)
        soft_min_dist = -torch.logsumexp(-distances / 0.1, dim=0) * 0.1
        
        # Combine both measures
        ged_approx = 0.5 * mahalanobis + 0.5 * soft_min_dist
        
        return ged_approx
    
    def _information_potential(self, point: torch.Tensor) -> torch.Tensor:
        """
        Compute information gain potential at a point
        
        Higher values where the point would add more information
        """
        # Compute similarities to all nodes
        similarities = F.cosine_similarity(
            point.unsqueeze(0),
            self.node_features,
            dim=1
        )
        
        # Information potential based on diversity
        # High when point is different from all existing nodes
        avg_similarity = similarities.mean()
        
        # Entropy of similarity distribution
        # High entropy = point relates to many nodes equally
        sim_probs = F.softmax(similarities / self.kT, dim=0)
        entropy = -(sim_probs * torch.log(sim_probs + 1e-8)).sum()
        
        # Combined information potential
        info_potential = (1 - avg_similarity) * entropy
        
        return info_potential
    
    def find_stable_point(
        self,
        start_point: torch.Tensor,
        learning_rate: float = 0.01,
        max_steps: int = 100,
        convergence_threshold: float = 1e-4
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Follow gradient to find stable point (local minimum)
        
        Args:
            start_point: Starting position (e.g., query vector)
            learning_rate: Step size for gradient descent
            max_steps: Maximum iterations
            convergence_threshold: Stop when gradient norm below this
            
        Returns:
            stable_point: Local minimum of geDIG field
            trajectory: List of field values during descent
        """
        current = start_point.clone()
        trajectory = []
        
        for step in range(max_steps):
            # Compute gradient
            grad = self.gradient(current.clone())
            
            # Check convergence
            grad_norm = grad.norm()
            if grad_norm < convergence_threshold:
                print(f"Converged at step {step}, grad_norm={grad_norm:.6f}")
                break
            
            # Gradient descent step (negative gradient for descent)
            current = current - learning_rate * grad
            
            # Optional: Project back to normalized sphere
            current = F.normalize(current, dim=0)
            
            # Record trajectory
            with torch.no_grad():
                field_value = self.gedig_field(current)
                trajectory.append(field_value.item())
            
            # Adaptive learning rate
            if step > 0 and step % 20 == 0:
                learning_rate *= 0.9
        
        return current, trajectory
    
    def find_critical_points(
        self,
        num_random_starts: int = 10,
        learning_rate: float = 0.01
    ) -> List[torch.Tensor]:
        """
        Find multiple critical points by starting from random positions
        
        Returns list of unique stable points
        """
        dim = self.node_features.size(1)
        critical_points = []
        
        for _ in range(num_random_starts):
            # Random starting point
            start = torch.randn(dim)
            start = F.normalize(start, dim=0)
            
            # Find stable point
            stable_point, _ = self.find_stable_point(start, learning_rate)
            
            # Check if it's a new critical point
            is_new = True
            for existing in critical_points:
                if torch.norm(stable_point - existing) < 0.1:
                    is_new = False
                    break
            
            if is_new:
                critical_points.append(stable_point)
        
        return critical_points
    
    def visualize_field_2d(
        self,
        projection_dims: Tuple[int, int] = (0, 1),
        resolution: int = 50
    ) -> np.ndarray:
        """
        Visualize 2D slice of the geDIG field
        
        Returns:
            2D array of field values for visualization
        """
        dim1, dim2 = projection_dims
        
        # Create grid
        x = torch.linspace(-2, 2, resolution)
        y = torch.linspace(-2, 2, resolution)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        
        # Evaluate field on grid
        field_values = torch.zeros(resolution, resolution)
        
        # Use mean of other dimensions
        base_point = self.feature_mean.clone()
        
        for i in range(resolution):
            for j in range(resolution):
                point = base_point.clone()
                point[dim1] = xx[i, j]
                point[dim2] = yy[i, j]
                
                with torch.no_grad():
                    field_values[i, j] = self.gedig_field(point)
        
        return field_values.numpy()


def test_gradient_field():
    """Test the gradient field approach"""
    # Create simple test graph
    num_nodes = 10
    dim = 64
    
    # Random node features (normalized)
    node_features = torch.randn(num_nodes, dim)
    node_features = F.normalize(node_features, dim=1)
    
    # Simple chain graph
    edge_index = torch.tensor([
        list(range(num_nodes-1)) + list(range(1, num_nodes)),
        list(range(1, num_nodes)) + list(range(num_nodes-1))
    ])
    
    graph = Data(x=node_features, edge_index=edge_index, num_nodes=num_nodes)
    
    # Create gradient field
    field = GeDIGGradientField(graph, w1=1.0, kT=0.1)
    
    # Test with random query
    query = torch.randn(dim)
    query = F.normalize(query, dim=0)
    
    print("Testing gradient field...")
    print(f"Initial field value: {field.gedig_field(query):.4f}")
    
    # Find stable point
    stable_point, trajectory = field.find_stable_point(query, learning_rate=0.05)
    
    print(f"\nTrajectory length: {len(trajectory)}")
    print(f"Initial value: {trajectory[0]:.4f}")
    print(f"Final value: {trajectory[-1]:.4f}")
    print(f"Improvement: {trajectory[0] - trajectory[-1]:.4f}")
    
    # Compare to nearest node
    nearest_dist = torch.norm(node_features - stable_point.unsqueeze(0), dim=1).min()
    print(f"\nDistance to nearest node: {nearest_dist:.4f}")
    
    # Find multiple critical points
    print("\nFinding critical points...")
    critical_points = field.find_critical_points(num_random_starts=5)
    print(f"Found {len(critical_points)} unique critical points")


if __name__ == "__main__":
    test_gradient_field()