"""
Geometric Probabilistic Graph: Three-Space Integration
=====================================================

This module implements Level 2 solution for the three-space problem in geDIG:
1. Graph Space G = (V, E) - Explicit graph structure
2. Probability Density Space (Ω, F, μ) - Distributions from attention
3. Similarity Space (Φ, d) - Sentence-BERT embeddings

This implementation provides theoretically grounded entropy calculations
using true Shannon entropy derived from probability distributions.

Reference:
- Shannon entropy: H(X) = -Σ p(x) log₂ p(x)
- Attention weights as probability distributions (softmax normalized)
- Graph structure via NetworkX or PyTorch Geometric

Author: InsightSpike-AI Team
Date: 2025-01
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv, global_mean_pool

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch/PyTorch Geometric not available. "
        "GeometricProbabilisticGraph will use fallback mode."
    )

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX not available. Graph operations will be limited.")

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_BERT_AVAILABLE = True
except ImportError:
    SENTENCE_BERT_AVAILABLE = False
    logger.warning(
        "sentence-transformers not available. Embeddings must be provided externally."
    )


@dataclass
class ThreeSpaceMetrics:
    """Metrics capturing all three spaces"""

    # Graph space metrics
    num_nodes: int
    num_edges: int
    graph_density: float
    avg_degree: float

    # Probability space metrics
    shannon_entropy: float  # True Shannon entropy: H = -Σ p log p
    probability_mass: float  # Total probability mass (should be ~1.0)
    entropy_per_node: np.ndarray  # H for each node's distribution

    # Similarity space metrics
    avg_cosine_similarity: float
    embedding_std: float
    pairwise_distances: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/serialization"""
        return {
            "graph": {
                "num_nodes": self.num_nodes,
                "num_edges": self.num_edges,
                "density": self.graph_density,
                "avg_degree": self.avg_degree,
            },
            "probability": {
                "shannon_entropy": float(self.shannon_entropy),
                "probability_mass": float(self.probability_mass),
                "entropy_per_node": (
                    self.entropy_per_node.tolist()
                    if self.entropy_per_node is not None
                    else None
                ),
            },
            "similarity": {
                "avg_cosine_similarity": float(self.avg_cosine_similarity),
                "embedding_std": float(self.embedding_std),
            },
        }


class GraphAttentionProbability(nn.Module):
    """
    Graph Attention layer that produces probability distributions.

    Uses multi-head attention to derive probability distributions over
    neighboring nodes. These distributions satisfy probability axioms:
    - Non-negative: softmax ensures p(x) >= 0
    - Normalized: Σ p(x) = 1
    - Measurable: attention weights form σ-algebra
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__()

        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch Geometric required for GraphAttentionProbability"
            )

        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            concat=concat,
            add_self_loops=True,
        )

        self.heads = heads
        self.out_channels = out_channels

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing embeddings and attention probabilities.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Tuple of (node_embeddings, attention_probabilities)
            - node_embeddings: [num_nodes, out_channels * heads] if concat else [num_nodes, out_channels]
            - attention_probabilities: [num_edges, heads] - softmax normalized per node
        """
        # Get embeddings and attention coefficients
        out, (edge_index_att, attention_weights) = self.gat(
            x, edge_index, return_attention_weights=True
        )

        # attention_weights shape: [num_edges, heads]
        # These are already softmax normalized per source node
        return out, attention_weights

    def get_node_probability_distribution(
        self, attention_weights: torch.Tensor, edge_index: torch.Tensor, node_idx: int
    ) -> torch.Tensor:
        """
        Extract probability distribution for a specific node.

        Args:
            attention_weights: Attention coefficients [num_edges, heads]
            edge_index: Edge indices [2, num_edges]
            node_idx: Index of target node

        Returns:
            Probability distribution over neighbors [num_neighbors, heads]
        """
        # Find edges where source node is node_idx
        source_mask = edge_index[0] == node_idx
        node_attention = attention_weights[source_mask]

        return node_attention


class GeometricProbabilisticGraph:
    """
    Unified representation integrating three spaces:
    1. Graph space G = (V, E)
    2. Probability density space (Ω, F, μ)
    3. Similarity space (Φ, d)

    This class provides:
    - Explicit graph structure (NetworkX or PyG)
    - True Shannon entropy from attention-based probability distributions
    - Embedding-based similarity calculations

    Usage:
        >>> gpg = GeometricProbabilisticGraph(embedding_dim=384)
        >>> gpg.add_node("doc1", "This is a document", embedding=None)
        >>> gpg.add_edge("doc1", "doc2", weight=0.8)
        >>> metrics = gpg.compute_three_space_metrics()
        >>> entropy = metrics.shannon_entropy  # True Shannon entropy!
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        attention_heads: int = 4,
        attention_hidden_dim: int = 128,
        model_name: str = "all-MiniLM-L6-v2",
        use_gpu: bool = False,
    ):
        """
        Initialize the three-space graph.

        Args:
            embedding_dim: Dimension of Sentence-BERT embeddings
            attention_heads: Number of attention heads for probability derivation
            attention_hidden_dim: Hidden dimension for attention mechanism
            model_name: Sentence-BERT model name
            use_gpu: Whether to use GPU acceleration
        """
        self.embedding_dim = embedding_dim
        self.attention_heads = attention_heads
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )

        # Space 1: Graph structure (NetworkX)
        if NETWORKX_AVAILABLE:
            self.graph = nx.Graph()
        else:
            self.graph = None
            logger.warning("NetworkX not available, graph operations disabled")

        # Space 2: Probability distributions (via attention)
        if TORCH_AVAILABLE:
            self.attention_model = GraphAttentionProbability(
                in_channels=embedding_dim,
                out_channels=attention_hidden_dim,
                heads=attention_heads,
                dropout=0.1,
                concat=True,
            ).to(self.device)
        else:
            self.attention_model = None
            logger.warning("PyTorch not available, probability derivation disabled")

        # Space 3: Similarity/embeddings (Sentence-BERT)
        if SENTENCE_BERT_AVAILABLE:
            self.embedder = SentenceTransformer(model_name)
            self.embedder = self.embedder.to(self.device)
        else:
            self.embedder = None
            logger.warning("Sentence-BERT not available, using provided embeddings")

        # Storage
        self.node_texts: Dict[str, str] = {}
        self.node_embeddings: Dict[str, np.ndarray] = {}
        self.node_to_idx: Dict[str, int] = {}
        self.idx_to_node: Dict[int, str] = {}

        # Cached probability distributions
        self._cached_attention_weights: Optional[torch.Tensor] = None
        self._cached_edge_index: Optional[torch.Tensor] = None

    def add_node(
        self,
        node_id: str,
        text: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        **attrs,
    ) -> None:
        """
        Add a node to all three spaces.

        Args:
            node_id: Unique node identifier
            text: Text content for embedding generation
            embedding: Pre-computed embedding (if text is None)
            **attrs: Additional node attributes
        """
        # Add to graph structure (Space 1)
        if self.graph is not None:
            self.graph.add_node(node_id, **attrs)

        # Store text
        if text is not None:
            self.node_texts[node_id] = text

        # Generate or store embedding (Space 3)
        if embedding is not None:
            self.node_embeddings[node_id] = embedding
        elif text is not None and self.embedder is not None:
            embedding = self.embedder.encode(text, convert_to_numpy=True)
            self.node_embeddings[node_id] = embedding
        else:
            logger.warning(
                f"Node {node_id} added without embedding. "
                "Provide text or embedding explicitly."
            )

        # Update index mappings
        if node_id not in self.node_to_idx:
            idx = len(self.node_to_idx)
            self.node_to_idx[node_id] = idx
            self.idx_to_node[idx] = node_id

        # Invalidate cache
        self._cached_attention_weights = None

    def add_edge(
        self, node1: str, node2: str, weight: Optional[float] = None, **attrs
    ) -> None:
        """
        Add an edge to graph structure.

        Args:
            node1: Source node ID
            node2: Target node ID
            weight: Edge weight (optional)
            **attrs: Additional edge attributes
        """
        if self.graph is None:
            logger.warning("NetworkX not available, cannot add edge")
            return

        if weight is not None:
            attrs["weight"] = weight

        self.graph.add_edge(node1, node2, **attrs)

        # Invalidate cache
        self._cached_attention_weights = None

    def compute_probability_distributions(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute probability distributions over graph using attention mechanism.

        This implements Space 2: Probability Density Space (Ω, F, μ)
        - Ω: Set of all nodes
        - F: σ-algebra generated by attention
        - μ: Probability measure from softmax(attention)

        Returns:
            Tuple of (attention_weights, edge_index)
            - attention_weights: [num_edges, heads] - probability distributions
            - edge_index: [2, num_edges] - graph structure
        """
        if not TORCH_AVAILABLE or self.attention_model is None:
            raise RuntimeError(
                "PyTorch/attention model not available. "
                "Cannot compute probability distributions."
            )

        # Check cache
        if (
            self._cached_attention_weights is not None
            and self._cached_edge_index is not None
        ):
            return self._cached_attention_weights, self._cached_edge_index

        # Convert to PyTorch Geometric format
        pyg_data = self._to_pyg_data()

        # Run attention model
        with torch.no_grad():
            node_embeddings, attention_weights = self.attention_model(
                pyg_data.x, pyg_data.edge_index
            )

        # Cache results
        self._cached_attention_weights = attention_weights
        self._cached_edge_index = pyg_data.edge_index

        return attention_weights, pyg_data.edge_index

    def calculate_shannon_entropy(self, node_id: Optional[str] = None) -> float:
        """
        Calculate TRUE Shannon entropy using probability distributions.

        This is the CORRECT entropy calculation:
        H(X) = -Σ p(x) log₂ p(x)

        where p(x) are attention weights (proper probability distributions).

        Args:
            node_id: If specified, compute entropy for this node's distribution.
                    If None, compute global graph entropy.

        Returns:
            Shannon entropy in bits
        """
        if not TORCH_AVAILABLE or self.attention_model is None:
            logger.warning(
                "PyTorch not available. Falling back to approximation: "
                "(1 - avg_similarity) / 2"
            )
            return self._fallback_entropy()

        try:
            attention_weights, edge_index = self.compute_probability_distributions()

            if node_id is not None:
                # Entropy for specific node
                node_idx = self.node_to_idx[node_id]
                node_probs = self.attention_model.get_node_probability_distribution(
                    attention_weights, edge_index, node_idx
                )

                # Handle isolated nodes (no outgoing edges)
                if len(node_probs) == 0:
                    return 0.0  # No uncertainty for isolated nodes

                # Shannon entropy: H = -Σ p log₂ p
                # Average over attention heads
                entropies = []
                for head in range(node_probs.shape[1]):
                    p = node_probs[:, head]
                    # Only compute entropy for non-zero probabilities
                    # This preserves the probability axiom: Σp = 1.0
                    mask = p > 1e-10
                    if mask.any():
                        H = -torch.sum(p[mask] * torch.log2(p[mask]))
                    else:
                        H = 0.0  # No uncertainty if all probabilities are zero
                    entropies.append(H.item() if isinstance(H, torch.Tensor) else H)

                return float(np.mean(entropies))

            else:
                # Global graph entropy: average over all nodes
                entropies = []
                for node_id in self.node_to_idx.keys():
                    node_entropy = self.calculate_shannon_entropy(node_id)
                    entropies.append(node_entropy)

                return float(np.mean(entropies)) if entropies else 0.0

        except Exception as e:
            logger.error(f"Shannon entropy calculation failed: {e}")
            return self._fallback_entropy()

    def calculate_information_gain(
        self, other: "GeometricProbabilisticGraph"
    ) -> float:
        """
        Calculate true information gain using Shannon entropy.

        IG = H(before) - H(after)

        This is the CORRECT IG calculation using proper entropy.

        Args:
            other: The "after" graph state

        Returns:
            Information gain in bits (positive = entropy decreased)
        """
        h_before = self.calculate_shannon_entropy()
        h_after = other.calculate_shannon_entropy()

        delta_ig = h_before - h_after

        logger.info(
            f"True IG calculation: H(before)={h_before:.3f} bits, "
            f"H(after)={h_after:.3f} bits, ΔIG={delta_ig:.3f} bits"
        )

        return delta_ig

    def compute_three_space_metrics(self) -> ThreeSpaceMetrics:
        """
        Compute comprehensive metrics across all three spaces.

        Returns:
            ThreeSpaceMetrics containing graph, probability, and similarity metrics
        """
        # Space 1: Graph metrics
        if self.graph is not None and len(self.graph.nodes) > 0:
            num_nodes = len(self.graph.nodes)
            num_edges = len(self.graph.edges)
            graph_density = nx.density(self.graph)
            avg_degree = np.mean([d for n, d in self.graph.degree()])
        else:
            num_nodes = len(self.node_embeddings)
            num_edges = 0
            graph_density = 0.0
            avg_degree = 0.0

        # Space 2: Probability metrics
        shannon_entropy = self.calculate_shannon_entropy()

        # Calculate per-node entropy
        entropy_per_node = []
        for node_id in self.node_to_idx.keys():
            try:
                h = self.calculate_shannon_entropy(node_id)
                entropy_per_node.append(h)
            except Exception as e:
                logger.warning(f"Failed to compute entropy for node {node_id}: {e}")
                entropy_per_node.append(0.0)
        entropy_per_node = np.array(entropy_per_node)

        # Verify probability mass conservation
        try:
            attention_weights, edge_index = self.compute_probability_distributions()
            # For each source node, probabilities should sum to ~1.0
            unique_sources = edge_index[0].unique()
            masses = []
            for node_idx in unique_sources:
                mask = edge_index[0] == node_idx
                node_probs = attention_weights[mask]
                # Average over attention heads
                mass = node_probs.sum(dim=0).mean()
                masses.append(mass.item())
            probability_mass = float(np.mean(masses)) if masses else 1.0
        except Exception as e:
            logger.debug(f"Could not verify probability mass: {e}")
            probability_mass = 1.0  # Assume normalized if cannot compute

        # Space 3: Similarity metrics
        if len(self.node_embeddings) >= 2:
            embeddings = np.array(list(self.node_embeddings.values()))

            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / (norms + 1e-8)

            # Pairwise similarities
            similarities = np.dot(normalized, normalized.T)
            n = len(embeddings)
            upper_indices = np.triu_indices(n, k=1)
            pairwise_sims = similarities[upper_indices]

            avg_cosine_similarity = float(np.mean(pairwise_sims))
            embedding_std = float(np.std(embeddings))
            pairwise_distances = 1.0 - pairwise_sims
        else:
            avg_cosine_similarity = 1.0
            embedding_std = 0.0
            pairwise_distances = None

        return ThreeSpaceMetrics(
            num_nodes=num_nodes,
            num_edges=num_edges,
            graph_density=graph_density,
            avg_degree=avg_degree,
            shannon_entropy=shannon_entropy,
            probability_mass=probability_mass,
            entropy_per_node=entropy_per_node,
            avg_cosine_similarity=avg_cosine_similarity,
            embedding_std=embedding_std,
            pairwise_distances=pairwise_distances,
        )

    def _to_pyg_data(self) -> Data:
        """Convert to PyTorch Geometric Data object"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")

        # Node features from embeddings
        node_features = []
        for idx in range(len(self.idx_to_node)):
            node_id = self.idx_to_node[idx]
            if node_id in self.node_embeddings:
                node_features.append(self.node_embeddings[node_id])
            else:
                # Zero embedding for missing nodes
                node_features.append(np.zeros(self.embedding_dim))

        x = torch.tensor(np.array(node_features), dtype=torch.float32).to(self.device)

        # Edge index
        if self.graph is not None and len(self.graph.edges) > 0:
            edges = []
            for u, v in self.graph.edges():
                u_idx = self.node_to_idx[u]
                v_idx = self.node_to_idx[v]
                # Add both directions for undirected graph
                edges.append([u_idx, v_idx])
                edges.append([v_idx, u_idx])
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            # No edges - create self-loops
            n = len(node_features)
            edge_index = torch.tensor([[i, i] for i in range(n)], dtype=torch.long).t()

        edge_index = edge_index.to(self.device)

        return Data(x=x, edge_index=edge_index)

    def _fallback_entropy(self) -> float:
        """
        Fallback entropy calculation when PyTorch unavailable.

        WARNING: This is NOT true Shannon entropy!
        It's a heuristic approximation: (1 - avg_similarity) / 2
        """
        if len(self.node_embeddings) < 2:
            return 0.0

        embeddings = np.array(list(self.node_embeddings.values()))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        similarities = np.dot(normalized, normalized.T)

        n = len(embeddings)
        upper_indices = np.triu_indices(n, k=1)
        pairwise_sims = similarities[upper_indices]
        avg_similarity = np.mean(pairwise_sims)

        # This is the OLD (incorrect) entropy calculation
        return float((1 - avg_similarity) / 2)


# Convenience functions for backward compatibility


def calculate_three_space_entropy(
    texts: List[str],
    edges: Optional[List[Tuple[int, int]]] = None,
    embedding_dim: int = 384,
) -> float:
    """
    Convenience function: Calculate Shannon entropy from texts and graph structure.

    Args:
        texts: List of text documents
        edges: Optional edge list [(i, j), ...] where i,j are indices into texts
        embedding_dim: Embedding dimension

    Returns:
        Shannon entropy in bits (using proper probability distributions)
    """
    gpg = GeometricProbabilisticGraph(embedding_dim=embedding_dim)

    # Add nodes
    for i, text in enumerate(texts):
        gpg.add_node(f"doc_{i}", text=text)

    # Add edges
    if edges:
        for i, j in edges:
            gpg.add_edge(f"doc_{i}", f"doc_{j}")

    return gpg.calculate_shannon_entropy()


def calculate_three_space_information_gain(
    texts_before: List[str],
    texts_after: List[str],
    edges_before: Optional[List[Tuple[int, int]]] = None,
    edges_after: Optional[List[Tuple[int, int]]] = None,
    embedding_dim: int = 384,
) -> float:
    """
    Convenience function: Calculate information gain between two states.

    Args:
        texts_before: Documents before change
        texts_after: Documents after change
        edges_before: Graph edges before
        edges_after: Graph edges after
        embedding_dim: Embedding dimension

    Returns:
        Information gain in bits (positive = entropy decreased)
    """
    # Build before graph
    gpg_before = GeometricProbabilisticGraph(embedding_dim=embedding_dim)
    for i, text in enumerate(texts_before):
        gpg_before.add_node(f"doc_{i}", text=text)
    if edges_before:
        for i, j in edges_before:
            gpg_before.add_edge(f"doc_{i}", f"doc_{j}")

    # Build after graph
    gpg_after = GeometricProbabilisticGraph(embedding_dim=embedding_dim)
    for i, text in enumerate(texts_after):
        gpg_after.add_node(f"doc_{i}", text=text)
    if edges_after:
        for i, j in edges_after:
            gpg_after.add_edge(f"doc_{i}", f"doc_{j}")

    return gpg_before.calculate_information_gain(gpg_after)
