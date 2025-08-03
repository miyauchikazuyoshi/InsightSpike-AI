"""
Vector Space geDIG Optimization
===============================

Find optimal vector positions in embedding space that minimize geDIG.
Instead of searching through discrete node combinations, directly derive
the optimal position in continuous vector space.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, get_laplacian


class VectorSpaceGeDIG:
    """Optimize geDIG directly in vector space"""
    
    def __init__(self, w1: float = 1.0, kT: float = 0.1):
        self.w1 = w1
        self.kT = kT
        
    def find_optimal_vector(
        self, 
        query_vector: torch.Tensor,
        current_graph: Data,
        method: str = "gradient"
    ) -> torch.Tensor:
        """
        Find optimal vector position that minimizes geDIG
        
        Args:
            query_vector: Query embedding
            current_graph: Current graph state
            method: "gradient" or "analytical"
            
        Returns:
            Optimal vector position
        """
        if method == "gradient":
            return self._gradient_descent_optimization(query_vector, current_graph)
        elif method == "analytical":
            return self._analytical_optimization(query_vector, current_graph)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _gradient_descent_optimization(
        self, 
        query_vector: torch.Tensor,
        current_graph: Data,
        lr: float = 0.01,
        max_iter: int = 100
    ) -> torch.Tensor:
        """Optimize using gradient descent"""
        # Initialize from query vector
        optimal_vec = query_vector.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([optimal_vec], lr=lr)
        
        for i in range(max_iter):
            # Compute geDIG-like energy function
            energy = self._compute_energy_function(optimal_vec, current_graph)
            
            # Backprop and update
            optimizer.zero_grad()
            energy.backward()
            optimizer.step()
            
            # Optional: Add regularization to prevent drift
            with torch.no_grad():
                # Keep vector normalized
                optimal_vec.data = F.normalize(optimal_vec.data, dim=0)
                
        return optimal_vec.detach()
    
    def _analytical_optimization(
        self,
        query_vector: torch.Tensor,
        current_graph: Data
    ) -> torch.Tensor:
        """Analytical solution using spectral decomposition"""
        # Get graph Laplacian
        edge_index = current_graph.edge_index
        num_nodes = current_graph.num_nodes
        
        # Compute normalized Laplacian
        edge_index, edge_weight = get_laplacian(
            edge_index, 
            normalization='sym',
            num_nodes=num_nodes
        )
        
        # Convert to dense for eigendecomposition
        L = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes)[0]
        
        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        
        # Project query into eigenspace
        query_proj = eigenvectors.T @ query_vector[:num_nodes]  # Truncate if needed
        
        # Compute optimal coefficients
        # Based on minimizing F = w1*ΔGED - kT*ΔIG
        optimal_coeffs = self._compute_optimal_coefficients(
            query_proj, eigenvalues
        )
        
        # Transform back to original space
        optimal_pos = eigenvectors @ optimal_coeffs
        
        # Extend to full dimension if needed
        if len(optimal_pos) < len(query_vector):
            # Pad with weighted average of existing nodes
            node_features = current_graph.x
            avg_features = node_features.mean(0)
            padding = avg_features[len(optimal_pos):]
            optimal_pos = torch.cat([optimal_pos, padding])
            
        return optimal_pos
    
    def _compute_energy_function(
        self,
        vector: torch.Tensor,
        graph: Data
    ) -> torch.Tensor:
        """
        Compute differentiable energy function
        F = w1*ΔGED - kT*ΔIG
        """
        # Compute similarity to existing nodes
        node_features = graph.x
        similarities = F.cosine_similarity(
            vector.unsqueeze(0), 
            node_features, 
            dim=1
        )
        
        # GED component: Distance from graph manifold
        # Use soft-min distance as proxy
        ged_component = -torch.logsumexp(-similarities / 0.1, dim=0) * 0.1
        
        # IG component: Information diversity
        # Use entropy of similarity distribution
        similarity_probs = F.softmax(similarities, dim=0)
        ig_component = -(similarity_probs * torch.log(similarity_probs + 1e-8)).sum()
        
        # Combined energy
        energy = self.w1 * ged_component - self.kT * ig_component
        
        return energy
    
    def _compute_optimal_coefficients(
        self,
        query_proj: torch.Tensor,
        eigenvalues: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute optimal coefficients in eigenspace
        """
        # Weight by inverse eigenvalues (smoother components)
        # Small eigenvalues = smooth eigenvectors = better for interpolation
        weights = 1.0 / (eigenvalues + 1e-6)
        weights = weights / weights.sum()
        
        # Optimal coefficients blend query projection with smoothness prior
        alpha = 0.7  # Balance between query similarity and smoothness
        optimal_coeffs = alpha * query_proj + (1 - alpha) * weights
        
        # Normalize
        optimal_coeffs = optimal_coeffs / optimal_coeffs.norm()
        
        return optimal_coeffs
    
    def compute_gedig_gradient(
        self,
        graph: Data,
        node_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute gradient of geDIG with respect to node positions
        Useful for understanding the optimization landscape
        """
        if node_idx is not None:
            # Gradient for specific node
            node_feature = graph.x[node_idx].requires_grad_(True)
        else:
            # Gradient for all nodes
            graph.x.requires_grad_(True)
            
        # Compute energy
        energy = self._compute_graph_energy(graph)
        
        # Get gradients
        energy.backward()
        
        if node_idx is not None:
            return node_feature.grad
        else:
            return graph.x.grad
    
    def _compute_graph_energy(self, graph: Data) -> torch.Tensor:
        """Compute total graph energy for gradient analysis"""
        # Simplified energy based on edge weights and node features
        edge_index = graph.edge_index
        node_features = graph.x
        
        # Edge energies
        src_features = node_features[edge_index[0]]
        dst_features = node_features[edge_index[1]]
        edge_energies = 1 - F.cosine_similarity(src_features, dst_features)
        
        # Node entropy
        feature_norms = node_features.norm(dim=1)
        node_entropy = -(feature_norms * torch.log(feature_norms + 1e-8)).mean()
        
        total_energy = edge_energies.mean() - self.kT * node_entropy
        
        return total_energy


def test_vector_optimization():
    """Test the vector space optimization"""
    # Create simple test graph
    num_nodes = 5
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3]
    ])
    
    # Random node features
    node_features = torch.randn(num_nodes, 128)
    node_features = F.normalize(node_features, dim=1)
    
    graph = Data(x=node_features, edge_index=edge_index, num_nodes=num_nodes)
    
    # Random query
    query = torch.randn(128)
    query = F.normalize(query, dim=0)
    
    # Initialize optimizer
    optimizer = VectorSpaceGeDIG(w1=1.0, kT=0.1)
    
    # Test gradient descent
    print("Testing gradient descent optimization...")
    optimal_gd = optimizer.find_optimal_vector(query, graph, method="gradient")
    print(f"Optimal vector norm: {optimal_gd.norm():.4f}")
    
    # Test analytical
    print("\nTesting analytical optimization...")
    optimal_analytical = optimizer.find_optimal_vector(query, graph, method="analytical")
    print(f"Optimal vector norm: {optimal_analytical.norm():.4f}")
    
    # Compare similarity to query
    sim_gd = F.cosine_similarity(query, optimal_gd, dim=0)
    sim_analytical = F.cosine_similarity(query, optimal_analytical, dim=0)
    
    print(f"\nSimilarity to query:")
    print(f"  Gradient descent: {sim_gd:.4f}")
    print(f"  Analytical: {sim_analytical:.4f}")


if __name__ == "__main__":
    test_vector_optimization()