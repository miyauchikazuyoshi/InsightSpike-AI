"""
PyTorch Geometric to NetworkX adapter for GED/IG calculations
"""

import logging
from typing import Any, Optional, Tuple
import numpy as np

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.utils import to_networkx

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class PyGAdapter:
    """Adapter to use PyTorch Geometric graphs with NetworkX-based algorithms"""

    @staticmethod
    def pyg_to_networkx(pyg_graph: Data) -> Any:
        """
        Convert PyTorch Geometric graph to NetworkX graph.

        Args:
            pyg_graph: PyTorch Geometric Data object

        Returns:
            NetworkX Graph object
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, returning empty graph")
            return nx.Graph() if NETWORKX_AVAILABLE else None

        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX not available, cannot convert")
            return None

        try:
            # Use built-in converter if available
            if hasattr(pyg_graph, "edge_index") and pyg_graph.edge_index.size(1) > 0:
                nx_graph = to_networkx(pyg_graph, to_undirected=True)
            else:
                # Create empty graph with nodes
                nx_graph = nx.Graph()
                if hasattr(pyg_graph, "num_nodes"):
                    nx_graph.add_nodes_from(range(pyg_graph.num_nodes))

            # Add node features if available
            if hasattr(pyg_graph, "x") and pyg_graph.x is not None:
                for i in range(pyg_graph.x.size(0)):
                    nx_graph.nodes[i]["feature"] = pyg_graph.x[i].cpu().numpy()

            # Add edge features if available
            if hasattr(pyg_graph, "edge_attr") and pyg_graph.edge_attr is not None:
                edge_index = pyg_graph.edge_index.cpu().numpy()
                for idx, (i, j) in enumerate(edge_index.T):
                    if nx_graph.has_edge(i, j):
                        nx_graph.edges[i, j]["weight"] = float(pyg_graph.edge_attr[idx])

            return nx_graph

        except Exception as e:
            logger.error(f"PyG to NetworkX conversion failed: {e}")
            # Return simple graph as fallback
            nx_graph = nx.Graph()
            if hasattr(pyg_graph, "num_nodes"):
                nx_graph.add_nodes_from(range(pyg_graph.num_nodes))
            return nx_graph

    @staticmethod
    def compute_embedding_distance(graph1: Data, graph2: Data) -> float:
        """
        Compute embedding-based distance between PyG graphs.

        Uses mean pooling of node features for graph-level comparison.
        """
        if not TORCH_AVAILABLE:
            return 0.0

        try:
            # Get node features
            x1 = graph1.x if hasattr(graph1, "x") and graph1.x is not None else None
            x2 = graph2.x if hasattr(graph2, "x") and graph2.x is not None else None

            if x1 is None or x2 is None:
                return 0.0

            # Mean pooling for graph embedding
            emb1 = x1.mean(dim=0).cpu().numpy()
            emb2 = x2.mean(dim=0).cpu().numpy()

            # Cosine distance
            similarity = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )
            distance = 1.0 - similarity

            return float(distance)

        except Exception as e:
            logger.error(f"Embedding distance calculation failed: {e}")
            return 0.0

    @staticmethod
    def extract_structural_features(pyg_graph: Data) -> dict:
        """Extract structural features from PyG graph for fast comparison"""
        features = {
            "num_nodes": 0,
            "num_edges": 0,
            "avg_degree": 0.0,
            "density": 0.0,
            "has_features": False,
        }

        if not TORCH_AVAILABLE or pyg_graph is None:
            return features

        try:
            # Basic counts
            features["num_nodes"] = (
                pyg_graph.num_nodes if hasattr(pyg_graph, "num_nodes") else 0
            )
            features["num_edges"] = (
                pyg_graph.edge_index.size(1) if hasattr(pyg_graph, "edge_index") else 0
            )

            # Average degree
            if features["num_nodes"] > 0 and features["num_edges"] > 0:
                features["avg_degree"] = (
                    2.0 * features["num_edges"] / features["num_nodes"]
                )

                # Graph density
                max_edges = features["num_nodes"] * (features["num_nodes"] - 1) / 2
                features["density"] = (
                    features["num_edges"] / max_edges if max_edges > 0 else 0.0
                )

            # Feature availability
            features["has_features"] = (
                hasattr(pyg_graph, "x") and pyg_graph.x is not None
            )

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")

        return features


class PyGGraphEditDistance:
    """
    Graph Edit Distance calculator that works directly with PyG graphs.

    Combines structural GED with embedding similarity for comprehensive distance.
    """

    def __init__(
        self,
        structural_weight: float = 0.7,
        embedding_weight: float = 0.3,
        use_networkx: bool = True,
    ):
        """
        Initialize PyG-compatible GED calculator.

        Args:
            structural_weight: Weight for structural GED component
            embedding_weight: Weight for embedding distance component
            use_networkx: Whether to use NetworkX for exact GED calculation
        """
        self.structural_weight = structural_weight
        self.embedding_weight = embedding_weight
        self.use_networkx = use_networkx and NETWORKX_AVAILABLE

        if self.use_networkx:
            from .graph_edit_distance import GraphEditDistance

            self.nx_ged_calculator = GraphEditDistance(optimization_level="standard")
        else:
            self.nx_ged_calculator = None

    def calculate(self, graph1: Data, graph2: Data) -> float:
        """
        Calculate combined GED between two PyG graphs.

        Returns:
            float: Combined distance score
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, returning fallback distance")
            return self._fallback_distance(graph1, graph2)

        # Structural distance
        if self.use_networkx and self.nx_ged_calculator:
            # Convert to NetworkX and use exact/approximate GED
            nx_g1 = PyGAdapter.pyg_to_networkx(graph1)
            nx_g2 = PyGAdapter.pyg_to_networkx(graph2)

            result = self.nx_ged_calculator.calculate(nx_g1, nx_g2)
            structural_distance = result.ged_value
        else:
            # Use fast approximation
            structural_distance = self._approximate_structural_distance(graph1, graph2)

        # Embedding distance
        embedding_distance = PyGAdapter.compute_embedding_distance(graph1, graph2)

        # Normalize distances
        max_nodes = max(
            graph1.num_nodes if hasattr(graph1, "num_nodes") else 1,
            graph2.num_nodes if hasattr(graph2, "num_nodes") else 1,
        )
        normalized_structural = structural_distance / max(max_nodes, 1)

        # Combine distances
        combined_distance = (
            self.structural_weight * normalized_structural
            + self.embedding_weight * embedding_distance
        )

        logger.debug(
            f"PyG GED: structural={structural_distance:.3f}, "
            f"embedding={embedding_distance:.3f}, "
            f"combined={combined_distance:.3f}"
        )

        return float(combined_distance)

    def compute_delta_ged(
        self,
        graph_before: Data,
        graph_after: Data,
        reference_graph: Optional[Data] = None,
    ) -> float:
        """
        Compute ΔGED for PyG graphs.

        Returns:
            float: Delta GED (negative indicates insight/simplification)
        """
        if reference_graph is not None:
            ged_before = self.calculate(graph_before, reference_graph)
            ged_after = self.calculate(graph_after, reference_graph)
            delta = ged_after - ged_before
        else:
            delta = self.calculate(graph_before, graph_after)

        return float(delta)

    def _approximate_structural_distance(self, graph1: Data, graph2: Data) -> float:
        """Fast approximation of structural distance"""
        feat1 = PyGAdapter.extract_structural_features(graph1)
        feat2 = PyGAdapter.extract_structural_features(graph2)

        # Simple structural difference
        node_diff = abs(feat1["num_nodes"] - feat2["num_nodes"])
        edge_diff = abs(feat1["num_edges"] - feat2["num_edges"])
        degree_diff = abs(feat1["avg_degree"] - feat2["avg_degree"])
        density_diff = abs(feat1["density"] - feat2["density"])

        # Weighted combination
        distance = (
            1.0 * node_diff
            + 0.5 * edge_diff / max(1, max(feat1["num_edges"], feat2["num_edges"]))
            + 0.3 * degree_diff
            + 0.2 * density_diff
        )

        return float(distance)

    def _fallback_distance(self, graph1: Any, graph2: Any) -> float:
        """Fallback when PyTorch not available"""
        try:
            # Try to extract basic info
            n1 = graph1.num_nodes if hasattr(graph1, "num_nodes") else 1
            n2 = graph2.num_nodes if hasattr(graph2, "num_nodes") else 1
            return float(abs(n2 - n1))
        except:
            return 1.0
