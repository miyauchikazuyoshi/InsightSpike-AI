"""
Structural Similarity Evaluation for Analogy Detection.

Provides methods to compare structural patterns of graphs/subgraphs,
enabling detection of analogies across different domains.

Example:
    Solar system (sun -> planets) ≈ Atom (nucleus -> electrons)
    Both share the "hub-and-spoke" structural pattern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from ..config.models import StructuralSimilarityConfig

logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """Result of structural similarity evaluation."""

    similarity: float
    method: str
    is_analogy: bool = False
    signature_a: Optional[np.ndarray] = None
    signature_b: Optional[np.ndarray] = None
    domain_a: Optional[str] = None
    domain_b: Optional[str] = None
    is_cross_domain: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


class StructuralSimilarityEvaluator:
    """Evaluates structural similarity between graphs/subgraphs.

    Supports multiple methods:
    - signature: Hop-growth pattern + topological features (default, fast)
    - spectral: Laplacian eigenvalue comparison (accurate, slower)
    - wl_kernel: Weisfeiler-Lehman kernel (requires grakel)
    - motif: Network motif frequency comparison

    Usage:
        config = StructuralSimilarityConfig(enabled=True, method="signature")
        evaluator = StructuralSimilarityEvaluator(config)
        result = evaluator.evaluate(graph1, graph2, center1="node_a", center2="node_b")
        if result.is_analogy:
            print(f"Analogy detected! Similarity: {result.similarity}")
    """

    def __init__(self, config: "StructuralSimilarityConfig"):
        """Initialize evaluator with configuration.

        Args:
            config: StructuralSimilarityConfig instance
        """
        self.config = config
        self._method_map = {
            "signature": self._compute_signature_similarity,
            "spectral": self._compute_spectral_similarity,
            "wl_kernel": self._compute_wl_similarity,
            "motif": self._compute_motif_similarity,
        }

    def evaluate(
        self,
        g1: nx.Graph,
        g2: nx.Graph,
        center1: Optional[str] = None,
        center2: Optional[str] = None,
    ) -> SimilarityResult:
        """Evaluate structural similarity between two graphs.

        Args:
            g1: First graph (or subgraph)
            g2: Second graph (or subgraph)
            center1: Optional center node for g1 (for ego-graph extraction)
            center2: Optional center node for g2 (for ego-graph extraction)

        Returns:
            SimilarityResult with similarity score and metadata
        """
        if not self.config.enabled:
            return SimilarityResult(
                similarity=0.0,
                method="disabled",
                is_analogy=False,
            )

        # Handle empty graphs
        if g1.number_of_nodes() == 0 or g2.number_of_nodes() == 0:
            return SimilarityResult(
                similarity=0.0,
                method=self.config.method,
                is_analogy=False,
                details={"reason": "empty_graph"},
            )

        # Get the appropriate method
        method_fn = self._method_map.get(self.config.method)
        if method_fn is None:
            logger.warning(f"Unknown method: {self.config.method}, falling back to signature")
            method_fn = self._compute_signature_similarity

        # Compute similarity
        result = method_fn(g1, g2, center1, center2)

        # Check cross-domain
        if self.config.cross_domain_only:
            result.is_cross_domain = self._is_cross_domain(g1, g2)
            result.is_analogy = (
                result.similarity >= self.config.analogy_threshold
                and result.is_cross_domain
            )
        else:
            result.is_analogy = result.similarity >= self.config.analogy_threshold

        return result

    def compute_analogy_bonus(
        self,
        g1: nx.Graph,
        g2: nx.Graph,
        center1: Optional[str] = None,
        center2: Optional[str] = None,
    ) -> float:
        """Compute analogy bonus for IG calculation.

        Returns analogy_weight * similarity if analogy is detected, else 0.

        Args:
            g1: First graph
            g2: Second graph
            center1: Optional center for g1
            center2: Optional center for g2

        Returns:
            Analogy bonus value (0 if no analogy detected)
        """
        result = self.evaluate(g1, g2, center1, center2)
        if result.is_analogy:
            bonus = self.config.analogy_weight * result.similarity
            logger.debug(
                f"[ANALOGY] similarity={result.similarity:.3f} "
                f"bonus={bonus:.3f} cross_domain={result.is_cross_domain}"
            )
            return bonus
        return 0.0

    # =========================================================================
    # Signature Method (Default)
    # =========================================================================

    def _compute_signature_similarity(
        self,
        g1: nx.Graph,
        g2: nx.Graph,
        center1: Optional[str],
        center2: Optional[str],
    ) -> SimilarityResult:
        """Compute similarity using hop-growth signature."""
        sig1 = self._extract_signature(g1, center1)
        sig2 = self._extract_signature(g2, center2)

        similarity = self._cosine_similarity(sig1, sig2)

        return SimilarityResult(
            similarity=similarity,
            method="signature",
            signature_a=sig1,
            signature_b=sig2,
            domain_a=self._get_dominant_domain(g1),
            domain_b=self._get_dominant_domain(g2),
            details={
                "signature_length": len(sig1),
                "g1_nodes": g1.number_of_nodes(),
                "g2_nodes": g2.number_of_nodes(),
            },
        )

    def _extract_signature(
        self,
        G: nx.Graph,
        center: Optional[str] = None,
    ) -> np.ndarray:
        """Extract structural signature from graph.

        The signature captures:
        - Node/edge counts at each hop
        - Density changes
        - Triangle counts (if enabled)
        - Clustering coefficients (if enabled)

        Args:
            G: Graph to extract signature from
            center: Optional center node for ego-graph

        Returns:
            Numpy array representing the structural signature
        """
        features: List[float] = []
        max_hops = self.config.max_signature_hops

        for hop in range(max_hops + 1):
            # Extract subgraph at this hop distance
            if center is not None and center in G:
                try:
                    sub = nx.ego_graph(G, center, radius=hop)
                except nx.NetworkXError:
                    sub = G
            else:
                sub = G

            n = sub.number_of_nodes()
            e = sub.number_of_edges()

            # Basic features: node count, edge count
            features.append(float(n))
            features.append(float(e))

            # Density
            if self.config.include_density:
                max_edges = n * (n - 1) / 2 if n > 1 else 1
                density = e / max_edges if max_edges > 0 else 0.0
                features.append(density)

            # Triangle count
            if self.config.include_triangles:
                if n >= 3:
                    try:
                        triangles = sum(nx.triangles(sub).values()) // 3
                        features.append(float(triangles))
                    except Exception:
                        features.append(0.0)
                else:
                    features.append(0.0)

            # Clustering coefficient
            if self.config.include_clustering:
                if n >= 2:
                    try:
                        clustering = nx.average_clustering(sub)
                        features.append(float(clustering))
                    except Exception:
                        features.append(0.0)
                else:
                    features.append(0.0)

            # Degree dispersion (helps separate hubs vs chains)
            if n > 0:
                try:
                    degrees = [deg for _, deg in sub.degree()]
                except Exception:
                    degrees = []
            else:
                degrees = []
            if degrees:
                degree_std = float(np.std(degrees))
                avg_deg = float(np.mean(degrees))
                max_deg = float(np.max(degrees))
                hub_ratio = max_deg / avg_deg if avg_deg > 0 else 0.0
            else:
                degree_std = 0.0
                hub_ratio = 0.0
            features.append(degree_std)
            features.append(hub_ratio)

            # Cycle rank (acyclic vs cyclic discrimination)
            if n > 0:
                try:
                    components = nx.number_connected_components(sub)
                except Exception:
                    components = 1
                cycle_rank = e - n + components
            else:
                cycle_rank = 0
            features.append(float(cycle_rank))

        return np.array(features, dtype=np.float32)

    # =========================================================================
    # Spectral Method
    # =========================================================================

    def _compute_spectral_similarity(
        self,
        g1: nx.Graph,
        g2: nx.Graph,
        center1: Optional[str],
        center2: Optional[str],
    ) -> SimilarityResult:
        """Compute similarity using Laplacian eigenvalues."""
        spec1 = self._extract_spectral_signature(g1, center1)
        spec2 = self._extract_spectral_signature(g2, center2)

        # Pad to same length
        max_len = max(len(spec1), len(spec2))
        spec1 = np.pad(spec1, (0, max_len - len(spec1)))
        spec2 = np.pad(spec2, (0, max_len - len(spec2)))

        # Compute similarity (inverse of normalized distance)
        distance = np.linalg.norm(spec1 - spec2)
        similarity = 1.0 / (1.0 + distance)

        return SimilarityResult(
            similarity=similarity,
            method="spectral",
            signature_a=spec1,
            signature_b=spec2,
            domain_a=self._get_dominant_domain(g1),
            domain_b=self._get_dominant_domain(g2),
            details={
                "spectral_distance": distance,
                "k": self.config.spectral_k,
            },
        )

    def _extract_spectral_signature(
        self,
        G: nx.Graph,
        center: Optional[str] = None,
    ) -> np.ndarray:
        """Extract spectral signature (top-k Laplacian eigenvalues)."""
        if center is not None and center in G:
            try:
                sub = nx.ego_graph(G, center, radius=self.config.max_signature_hops)
            except nx.NetworkXError:
                sub = G
        else:
            sub = G

        if sub.number_of_nodes() < 2:
            return np.zeros(self.config.spectral_k, dtype=np.float32)

        try:
            L = nx.laplacian_matrix(sub).toarray().astype(np.float64)
            eigenvalues = np.linalg.eigvalsh(L)
            # Take top-k smallest non-zero eigenvalues (sorted)
            eigenvalues = np.sort(eigenvalues)
            k = min(self.config.spectral_k, len(eigenvalues))
            return eigenvalues[:k].astype(np.float32)
        except Exception as e:
            logger.warning(f"Spectral extraction failed: {e}")
            return np.zeros(self.config.spectral_k, dtype=np.float32)

    # =========================================================================
    # WL Kernel Method (requires grakel)
    # =========================================================================

    def _compute_wl_similarity(
        self,
        g1: nx.Graph,
        g2: nx.Graph,
        center1: Optional[str],
        center2: Optional[str],
    ) -> SimilarityResult:
        """Compute similarity using Weisfeiler-Lehman kernel."""
        try:
            from grakel import Graph as GrakelGraph
            from grakel.kernels import WeisfeilerLehman, VertexHistogram
        except ImportError:
            logger.warning("grakel not installed, falling back to signature method")
            return self._compute_signature_similarity(g1, g2, center1, center2)

        # Extract subgraphs if centers provided
        if center1 is not None and center1 in g1:
            g1 = nx.ego_graph(g1, center1, radius=self.config.max_signature_hops)
        if center2 is not None and center2 in g2:
            g2 = nx.ego_graph(g2, center2, radius=self.config.max_signature_hops)

        # Convert to grakel format
        def nx_to_grakel(G: nx.Graph) -> GrakelGraph:
            # Create adjacency dict
            adj = {n: list(G.neighbors(n)) for n in G.nodes()}
            # Node labels (use degree as default label)
            labels = {n: G.degree(n) for n in G.nodes()}
            return GrakelGraph(adj, node_labels=labels)

        try:
            gk1 = nx_to_grakel(g1)
            gk2 = nx_to_grakel(g2)

            # WL kernel
            wl = WeisfeilerLehman(
                n_iter=self.config.wl_iterations,
                base_graph_kernel=VertexHistogram,
                normalize=True,
            )
            K = wl.fit_transform([gk1, gk2])
            similarity = float(K[0, 1])

            return SimilarityResult(
                similarity=similarity,
                method="wl_kernel",
                domain_a=self._get_dominant_domain(g1),
                domain_b=self._get_dominant_domain(g2),
                details={
                    "wl_iterations": self.config.wl_iterations,
                },
            )
        except Exception as e:
            logger.warning(f"WL kernel failed: {e}, falling back to signature")
            return self._compute_signature_similarity(g1, g2, center1, center2)

    # =========================================================================
    # Motif Method
    # =========================================================================

    def _compute_motif_similarity(
        self,
        g1: nx.Graph,
        g2: nx.Graph,
        center1: Optional[str],
        center2: Optional[str],
    ) -> SimilarityResult:
        """Compute similarity using network motif frequencies."""
        motif1 = self._extract_motif_signature(g1, center1)
        motif2 = self._extract_motif_signature(g2, center2)

        weights = self._motif_feature_weights(len(motif1))
        motif1_weighted = motif1 * weights
        motif2_weighted = motif2 * weights

        raw_similarity = self._cosine_similarity(motif1_weighted, motif2_weighted)
        penalty = self._motif_penalty(motif1, motif2)
        similarity = raw_similarity * penalty

        return SimilarityResult(
            similarity=similarity,
            method="motif",
            signature_a=motif1,
            signature_b=motif2,
            domain_a=self._get_dominant_domain(g1),
            domain_b=self._get_dominant_domain(g2),
            details={
                "motif_counts_g1": motif1.tolist(),
                "motif_counts_g2": motif2.tolist(),
                "motif_weights": weights.tolist(),
                "motif_similarity_raw": raw_similarity,
                "motif_penalty": penalty,
            },
        )

    def _motif_feature_weights(self, length: int) -> np.ndarray:
        """Return weights for motif features."""
        base = np.array(
            [
                1.0,  # triangles
                1.0,  # stars
                1.2,  # hub_ratio
                1.0,  # chains
                2.0,  # squares
                0.4,  # density
                1.0,  # degree_cv
                1.6,  # cycle_rank
                0.4,  # avg_path_norm
                0.4,  # diameter_norm
                1.2,  # leaves
            ],
            dtype=np.float32,
        )
        if length <= 0:
            return np.zeros(0, dtype=np.float32)
        if length == len(base):
            return base
        if length < len(base):
            return base[:length]
        return np.pad(base, (0, length - len(base)), constant_values=1.0)

    def _motif_penalty(self, motif1: np.ndarray, motif2: np.ndarray) -> float:
        """Down-weight mismatched motifs that look similar by cosine alone."""
        if len(motif1) < 5 or len(motif2) < 5:
            return 1.0

        penalty = 1.0

        hub1 = float(motif1[2])
        hub2 = float(motif2[2])
        if hub1 > 0 and hub2 > 0:
            ratio = max(hub1, hub2) / max(min(hub1, hub2), 1e-6)
            if ratio > 2.0:
                penalty *= 1.0 / ratio

        squares1 = float(motif1[4])
        squares2 = float(motif2[4])
        if (squares1 < 0.05 and squares2 > 0.15) or (squares2 < 0.05 and squares1 > 0.15):
            penalty *= 0.7

        return penalty

    def _extract_motif_signature(
        self,
        G: nx.Graph,
        center: Optional[str] = None,
    ) -> np.ndarray:
        """Extract motif frequency signature.

        Counts common motifs and topology cues:
        - Triangles (3-clique)
        - Stars (1 hub + k spokes)
        - Chains/paths
        - Squares (4-cycle, chordless)
        - Cycle rank, degree dispersion, path length stats
        """
        if center is not None and center in G:
            try:
                sub = nx.ego_graph(G, center, radius=self.config.max_signature_hops)
            except nx.NetworkXError:
                sub = G
        else:
            sub = G

        n = sub.number_of_nodes()
        e = sub.number_of_edges()

        # Triangle count
        if n >= 3:
            try:
                triangles = sum(nx.triangles(sub).values()) // 3
            except Exception:
                triangles = 0
        else:
            triangles = 0

        # Star count (nodes with degree >= 3)
        degrees = dict(sub.degree())
        stars = sum(1 for d in degrees.values() if d >= 3)
        leaves = sum(1 for d in degrees.values() if d == 1)

        # Hub-spoke ratio (max degree / avg degree)
        if degrees and n > 1:
            max_deg = max(degrees.values())
            avg_deg = sum(degrees.values()) / n
            hub_ratio = max_deg / avg_deg if avg_deg > 0 else 0
        else:
            hub_ratio = 0

        # Chain detection (nodes with degree == 2)
        chains = sum(1 for d in degrees.values() if d == 2)

        # Square (4-cycle) count using non-adjacent node pairs
        try:
            squares = 0
            if n >= 4:
                neighbors = {node: set(sub.neighbors(node)) for node in sub.nodes()}
                nodes = list(neighbors.keys())
                for idx, u in enumerate(nodes):
                    for v in nodes[idx + 1:]:
                        if v in neighbors[u]:
                            continue
                        common = neighbors[u] & neighbors[v]
                        if len(common) >= 2:
                            squares += (len(common) * (len(common) - 1)) // 2
                squares = squares // 2
        except Exception:
            squares = 0

        # Degree dispersion (coefficient of variation)
        if degrees:
            deg_values = list(degrees.values())
            degree_mean = float(np.mean(deg_values))
            degree_std = float(np.std(deg_values))
            degree_cv = degree_std / degree_mean if degree_mean > 0 else 0.0
        else:
            degree_cv = 0.0

        # Cycle rank (acyclic vs cyclic discrimination)
        if n > 0:
            try:
                components = nx.number_connected_components(sub)
            except Exception:
                components = 1
            cycle_rank = e - n + components
        else:
            cycle_rank = 0

        # Path length stats (normalized by graph size)
        avg_path_norm = 0.0
        diameter_norm = 0.0
        if n > 1:
            try:
                if nx.is_connected(sub):
                    avg_path = nx.average_shortest_path_length(sub)
                    diameter = nx.diameter(sub)
                    denom = max(n - 1, 1)
                else:
                    largest_nodes = max(nx.connected_components(sub), key=len)
                    largest = sub.subgraph(largest_nodes)
                    if largest.number_of_nodes() > 1:
                        avg_path = nx.average_shortest_path_length(largest)
                        diameter = nx.diameter(largest)
                        denom = max(largest.number_of_nodes() - 1, 1)
                    else:
                        avg_path = 0.0
                        diameter = 0.0
                        denom = 1
                avg_path_norm = avg_path / denom
                diameter_norm = diameter / denom
            except Exception:
                avg_path_norm = 0.0
                diameter_norm = 0.0

        # Normalize by graph size
        norm = max(n, 1)
        return np.array([
            triangles / norm,
            stars / norm,
            hub_ratio,
            chains / norm,
            squares / norm,
            e / max(n * (n - 1) / 2, 1),  # density
            degree_cv,
            cycle_rank / norm,
            avg_path_norm,
            diameter_norm,
            leaves / norm,
        ], dtype=np.float32)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) == 0 or len(b) == 0:
            return 0.0

        # Ensure same length
        if len(a) != len(b):
            max_len = max(len(a), len(b))
            a = np.pad(a, (0, max_len - len(a)))
            b = np.pad(b, (0, max_len - len(b)))

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def _is_cross_domain(self, g1: nx.Graph, g2: nx.Graph) -> bool:
        """Check if two graphs belong to different domains."""
        domain1 = self._get_dominant_domain(g1)
        domain2 = self._get_dominant_domain(g2)

        if domain1 is None or domain2 is None:
            # If domain info not available, assume cross-domain
            return True

        return domain1 != domain2

    def _get_dominant_domain(self, G: nx.Graph) -> Optional[str]:
        """Get the dominant domain from node attributes."""
        attr_name = self.config.domain_attribute
        domains: Dict[str, int] = {}

        for node in G.nodes():
            data = G.nodes[node]
            domain = data.get(attr_name) or data.get("category")
            if domain:
                domains[domain] = domains.get(domain, 0) + 1

        if not domains:
            return None

        return max(domains, key=domains.get)  # type: ignore


# Convenience function for quick evaluation
def compute_structural_similarity(
    g1: nx.Graph,
    g2: nx.Graph,
    method: str = "signature",
    threshold: float = 0.7,
) -> SimilarityResult:
    """Quick structural similarity computation.

    Args:
        g1: First graph
        g2: Second graph
        method: Similarity method (signature, spectral, wl_kernel, motif)
        threshold: Analogy detection threshold

    Returns:
        SimilarityResult
    """
    from ..config.models import StructuralSimilarityConfig

    config = StructuralSimilarityConfig(
        enabled=True,
        method=method,  # type: ignore
        analogy_threshold=threshold,
        cross_domain_only=False,
    )
    evaluator = StructuralSimilarityEvaluator(config)
    return evaluator.evaluate(g1, g2)


__all__ = [
    "StructuralSimilarityEvaluator",
    "SimilarityResult",
    "compute_structural_similarity",
]
